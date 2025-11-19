import argparse
import tempfile
import torch
import warnings
import time
import logging
import threading
import numpy as np
import PIL.Image
from pathlib import Path
from typing import Optional, Union
from contextlib import contextmanager

warnings.filterwarnings("ignore")
import os
import trimesh
try:
    import meshoptimizer as mopt
except ImportError:  # pragma: no cover - optional dependency
    mopt = None

from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
from step1x3d_geometry.models.pipelines.pipeline_utils import (
    reduce_face,
    remove_degenerate_face,
    preprocess_image,
)

logger = logging.getLogger(__name__)


SUPPORTED_FORMATS = {"glb", "obj", "fbx"}
BYTES_IN_GB = 1024 ** 3


def export_mesh(mesh: "trimesh.Trimesh", output_path: str, output_format: Optional[str] = None) -> None:
    """
    Export mesh as GLB/OBJ via trimesh or FBX via pyassimp.
    """
    if not output_path:
        raise ValueError("output_path must be provided.")

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    fmt = (output_format or os.path.splitext(output_path)[1].lstrip(".")).lower()
    if not fmt:
        raise ValueError("Unable to infer export format. Provide --output-format or a file extension.")
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported export format: {fmt}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")

    if fmt in {"glb", "obj"}:
        mesh.export(output_path, file_type=fmt)
        return

    if fmt == "fbx":
        try:
            import pyassimp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("FBX export requires pyassimp. Install it via `pip install pyassimp`.") from exc

        fd, temp_path = tempfile.mkstemp(suffix=".obj")
        os.close(fd)
        try:
            mesh.export(temp_path, file_type="obj")
            scene = pyassimp.load(temp_path)
            try:
                pyassimp.export(scene, output_path, file_type="fbx")
            finally:
                pyassimp.release(scene)
        finally:
            os.remove(temp_path)
        return

    raise ValueError(f"Unsupported export format: {fmt}")


# def run_background_removal_stage(input_image_path: str, rembg_backend: str, use_rembg_cache: bool) -> PIL.Image.Image:
#     """
#     Run rembg-based background removal and preprocessing as a standalone stage.
#     """
#     if not input_image_path:
#         raise ValueError("input_image_path must be provided for background removal.")
#     stage_start = time.perf_counter()
#     with PIL.Image.open(input_image_path) as image:
#         source = image.convert("RGBA")
#     processed = preprocess_image(
#         source,
#         force=False,
#         background_color=[255, 255, 255],
#         foreground_ratio=0.95,
#         rembg_backend=rembg_backend,
#         use_rembg_cache=use_rembg_cache,
#     )
#     logger.info("Background removal finished in %.2fs", time.perf_counter() - stage_start)
#     return processed


def run_geometry_model_stage(
    input_image: Union[str, PIL.Image.Image],
    # rembg_backend: str,
    # preprocess_in_pipeline: bool,
    # use_rembg_cache: bool,
) -> tuple["trimesh.Trimesh", Union[str, torch.device]]:
    """
    Run the Step1X model and return the raw mesh as well as the pipeline device.
    """
    stage_start = time.perf_counter()
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder="Step1X-3D-Geometry-1300m"
    ).to("cuda")

    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(2025)
    out = pipeline(
        input_image,
        guidance_scale=7.5,
        num_inference_steps=50,
        generator=generator,
        # rembg_backend=rembg_backend,
        # preprocess_input=preprocess_in_pipeline,
        # use_rembg_cache=use_rembg_cache,
    )
    duration = time.perf_counter() - stage_start
    logger.info("Geometry model stage finished in %.2fs", duration)
    return out.mesh[0], pipeline.device


def refine_and_optimize_mesh(mesh: "trimesh.Trimesh", max_facenum: int = 50000) -> "trimesh.Trimesh":
    """
    Apply pymeshlab-based cleanup followed by meshoptimizer passes.
    """
    cleaned = remove_degenerate_face(mesh)
    cleaned = reduce_face(cleaned, max_facenum=max_facenum)

    if mopt is None:
        return cleaned

    vertices = np.array(cleaned.vertices, dtype=np.float32, copy=True)
    faces = np.array(cleaned.faces, dtype=np.uint32, copy=True)
    if vertices.size == 0 or faces.size == 0:
        return cleaned

    indices = faces.reshape(-1)
    vertex_count = vertices.shape[0]
    index_count = indices.shape[0]

    try:
        remap = np.empty(vertex_count, dtype=np.uint32)
        mopt.generate_vertex_remap(
            remap,
            indices,
            index_count=index_count,
            vertices=vertices,
            vertex_count=vertex_count,
            vertex_size=vertices.strides[0],
        )

        remapped_indices = np.empty_like(indices)
        mopt.remap_index_buffer(remapped_indices, indices, index_count=index_count, remap=remap)

        remapped_vertices = np.empty_like(vertices)
        mopt.remap_vertex_buffer(
            remapped_vertices,
            vertices,
            vertex_count=vertex_count,
            vertex_size=vertices.strides[0],
            remap=remap,
        )

        cache_optimized = np.empty_like(remapped_indices)
        mopt.optimize_vertex_cache(
            cache_optimized, remapped_indices, index_count=index_count, vertex_count=vertex_count
        )

        overdraw_optimized = np.empty_like(cache_optimized)
        mopt.optimize_overdraw(
            overdraw_optimized,
            cache_optimized,
            remapped_vertices[:, :3],
            index_count=index_count,
            vertex_count=vertex_count,
            vertex_positions_stride=remapped_vertices.strides[0],
            threshold=1.05,
        )

        fetched_vertices = np.empty_like(remapped_vertices)
        fetch_indices = np.copy(overdraw_optimized)
        mopt.optimize_vertex_fetch(
            fetched_vertices,
            fetch_indices,
            remapped_vertices,
            index_count=index_count,
            vertex_count=vertex_count,
            vertex_size=remapped_vertices.strides[0],
        )
    except Exception as exc:  # pragma: no cover - best effort optimization
        logger.warning("meshoptimizer refinement failed (%s); using cleaned mesh.", exc)
        return cleaned

    faces_opt = fetch_indices.reshape(-1, 3).astype(np.int64, copy=False)
    optimized = trimesh.Trimesh(vertices=fetched_vertices, faces=faces_opt, process=False)
    optimized.metadata.update(getattr(cleaned, "metadata", {}))
    return optimized


def basic_mesh_cleanup_stage(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    Apply baseline cleanups and meshoptimizer refinement.
    """
    cleanup_start = time.perf_counter()
    mesh = refine_and_optimize_mesh(mesh)
    logger.info("Mesh cleanup stage finished in %.2fs", time.perf_counter() - cleanup_start)
    return mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step-1X geometry inference and export to GLB/OBJ/FBX.")
    parser.add_argument("--input", default="examples/imgs/captured.jpeg", help="Input image path.")
    parser.add_argument("--output-dir", default="output", help="Directory used to store exported mesh.")
    parser.add_argument("--output-name", default="captured", help="Filename stem (without extension) for the mesh.")
    parser.add_argument(
        "--output-format",
        default="glb",
        choices=sorted(SUPPORTED_FORMATS),
        help="Export format applied to the generated mesh.",
    )
    parser.add_argument("--auto-uv", action="store_true", help="Enable automatic UV unwrapping for the generated mesh.")
    parser.add_argument("--uv-texture-size", type=int, default=1024, help="Texture map resolution for UV layout.")
    parser.add_argument("--uv-render-size", type=int, default=512, help="Render size passed to UVProjection.")
    parser.add_argument(
        "--uv-sampling-mode",
        choices=["nearest", "bilinear"],
        default="nearest",
        help="Sampling mode used when generating UV atlases.",
    )
    parser.add_argument(
        "--uv-device",
        default="cuda",
        help="Device used for UV generation (defaults to pipeline device, e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--rembg-backend",
        default="birefnet-general",
        help="Background removal backend passed to Step1X geometry pipeline.",
    )
    parser.add_argument(
        "--disable-rembg",
        action="store_true",
        help="Skip the standalone rembg preprocessing stage and rely on the pipeline defaults.",
    )
    parser.add_argument(
        "--disable-rembg-cache",
        action="store_true",
        help="Disable rembg session caching (forces a fresh ONNX Runtime session per call).",
    )
    return parser.parse_args()


def resolve_output_path(output_dir: str, output_name: str, output_format: str) -> str:
    if not output_format:
        raise ValueError("output_format must be provided to construct the output path.")
    filename = f"{output_name}.{output_format}"
    return str((Path(output_dir) / filename).resolve())


def generate_uv_ready_mesh(
    mesh: "trimesh.Trimesh",
    texture_size: int,
    render_size: int,
    sampling_mode: str,
    uv_device: Union[str, torch.device],
) -> "trimesh.Trimesh":
    uv_start = time.perf_counter()
    try:
        from step1x3d_texture.texture_sync.project import UVProjection
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Auto-UV generation requires the texture_sync dependencies (pytorch3d, xatlas). "
            "Install pytorch3d as documented before using --auto-uv."
        ) from exc

    device = torch.device(uv_device)
    uv_projection = UVProjection(
        texture_size=texture_size,
        render_size=render_size,
        sampling_mode=sampling_mode,
        channels=3,
        device=device,
    )
    uv_projection.load_mesh(mesh, autouv=True)

    texture_map = torch.ones(
        (texture_size, texture_size, 3), dtype=torch.float32, device=device
    ).cpu()

    fd, temp_path = tempfile.mkstemp(suffix=".obj")
    os.close(fd)
    try:
        uv_projection.save_mesh(temp_path, texture_map)
        uv_mesh = trimesh.load(temp_path, force="mesh")
    finally:
        os.remove(temp_path)

    uv_duration = time.perf_counter() - uv_start
    logger.info("UV mapping took %.2fs", uv_duration)
    return uv_mesh


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class StageMemoryMonitor:
    """
    Lightweight helper that samples overall GPU memory usage via NVML and prints per-stage summaries.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self._nvml = None
        self._nvml_handle = None
        self._nvml_sample_interval = 0.05
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml
        except ModuleNotFoundError:
            return
        try:
            pynvml.nvmlInit()
            device_index = torch.cuda.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            return
        self._nvml = pynvml
        self._nvml_handle = handle

    class _NVMLSampler:
        def __init__(self, read_fn, interval: float) -> None:
            self._read_fn = read_fn
            self._interval = interval
            self._lock = threading.Lock()
            self._running = False
            self._thread: Optional[threading.Thread] = None
            self._max_used = 0.0

        def start(self, initial_used: float) -> None:
            self._max_used = initial_used
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def _run(self) -> None:
            while self._running:
                try:
                    used = self._read_fn()
                except Exception:
                    break
                with self._lock:
                    if used > self._max_used:
                        self._max_used = used
                time.sleep(self._interval)

        def stop(self) -> None:
            if not self._running:
                return
            self._running = False
            if self._thread is not None:
                self._thread.join()
                self._thread = None

        def max_used(self) -> float:
            with self._lock:
                return self._max_used

    def _read_nvml_memory(self) -> Optional[tuple[float, float]]:
        if self._nvml is None or self._nvml_handle is None:
            return None
        try:
            info = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        except Exception:
            return None
        return float(info.used), float(info.total)

    def _read_nvml_used_value(self) -> float:
        memory = self._read_nvml_memory()
        return memory[0] if memory is not None else 0.0

    @contextmanager
    def track(self, label: str):
        stage_start = time.perf_counter()
        nvml_snapshot: Optional[dict[str, float]] = None
        nvml_sampler: Optional[StageMemoryMonitor._NVMLSampler] = None
        if self._nvml is not None:
            nvml_memory = self._read_nvml_memory()
            if nvml_memory is not None:
                used_bytes, total_bytes = nvml_memory
                nvml_snapshot = {"base_used": used_bytes, "total": total_bytes}
                nvml_sampler = StageMemoryMonitor._NVMLSampler(
                    self._read_nvml_used_value,
                    self._nvml_sample_interval,
                )
                nvml_sampler.start(used_bytes)
        try:
            yield
        finally:
            self._record(label, stage_start, nvml_snapshot, nvml_sampler)

    def _record(
        self,
        label: str,
        stage_start: float,
        nvml_snapshot: Optional[dict[str, float]],
        nvml_sampler: Optional[_NVMLSampler],
    ) -> None:
        duration = time.perf_counter() - stage_start
        nvml_record: Optional[dict[str, float]] = None
        if nvml_snapshot is not None:
            if nvml_sampler is not None:
                nvml_sampler.stop()
                sampler_peak = nvml_sampler.max_used()
            else:
                sampler_peak = nvml_snapshot["base_used"]
            current_memory = self._read_nvml_memory()
            if current_memory is not None:
                residual_used, total_bytes = current_memory
            else:
                residual_used = nvml_snapshot["base_used"]
                total_bytes = nvml_snapshot["total"]
            peak_used = max(sampler_peak, residual_used)
            base_used = nvml_snapshot["base_used"]
            delta_used = max(0.0, peak_used - base_used)
            nvml_record = {
                "base_used_gb": base_used / BYTES_IN_GB,
                "peak_used_gb": peak_used / BYTES_IN_GB,
                "delta_used_gb": delta_used / BYTES_IN_GB,
                "residual_used_gb": residual_used / BYTES_IN_GB,
                "total_gb": total_bytes / BYTES_IN_GB,
            }
        self.records.append({"label": label, "duration": duration, "nvml": nvml_record})

    def report(self) -> None:
        if not self.records:
            logger.info("No GPU memory stats captured.")
            return
        logger.info("Per-stage GPU memory summary (NVML):")
        for record in self.records:
            nvml_record = record["nvml"]
            if nvml_record is None:
                msg = "NVML stats unavailable"
            else:
                msg = (
                    "base=%.2fGB peak=%.2fGB delta=%.2fGB residual=%.2fGB / total=%.2fGB"
                    % (
                        nvml_record["base_used_gb"],
                        nvml_record["peak_used_gb"],
                        nvml_record["delta_used_gb"],
                        nvml_record["residual_used_gb"],
                        nvml_record["total_gb"],
                    )
                )
            logger.info("%s | duration=%.2fs | %s", record["label"], record["duration"], msg)


def log_mesh_stats(mesh: "trimesh.Trimesh", label: str) -> None:
    if not isinstance(mesh, trimesh.Trimesh):
        logger.info("%s: mesh stats unavailable for type %s", label, type(mesh))
        return
    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.faces)
    bounds = getattr(mesh, "bounds", None)
    if bounds is not None:
        min_corner = tuple(bounds[0].tolist())
        max_corner = tuple(bounds[1].tolist())
        extents = tuple((bounds[1] - bounds[0]).tolist())
    else:
        min_corner = max_corner = extents = None
    logger.info(
        "%s: %d vertices, %d faces, bbox_min=%s, bbox_max=%s, extents=%s",
        label,
        num_vertices,
        num_faces,
        min_corner,
        max_corner,
        extents,
    )


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    output_path = resolve_output_path(args.output_dir, args.output_name, args.output_format)
    memory_monitor = StageMemoryMonitor()
    logger.info(
        "Starting geometry generation | input=%s output=%s format=%s enable_uv=%s rembg_backend=%s",
        args.input,
        output_path,
        args.output_format,
        args.auto_uv,
        args.rembg_backend,
    )
    try:
        processed_input: Union[str, PIL.Image.Image] = args.input
        # preprocess_in_pipeline = True
        # use_rembg_cache = not args.disable_rembg_cache
        # if not args.disable_rembg:
        #     with memory_monitor.track("Stage: background removal"):
        #         processed_input = run_background_removal_stage(
        #             input_image_path=args.input,
        #             rembg_backend=args.rembg_backend,
        #             use_rembg_cache=use_rembg_cache,
        #         )
        #     preprocess_in_pipeline = False

        with memory_monitor.track("Stage: geometry model"):
            mesh, pipeline_device = run_geometry_model_stage(
                input_image=processed_input,
                # rembg_backend=args.rembg_backend,
                # preprocess_in_pipeline=preprocess_in_pipeline,
                # use_rembg_cache=use_rembg_cache,
            )
        log_mesh_stats(mesh, "Mesh from geometry model")

        with memory_monitor.track("Stage: mesh cleanup"):
            mesh = basic_mesh_cleanup_stage(mesh)
        log_mesh_stats(mesh, "Mesh after cleanup")

        if args.auto_uv:
            target_device = args.uv_device or str(pipeline_device)
            logger.info(
                "UV config | texture_size=%d render_size=%d sampling=%s device=%s",
                args.uv_texture_size,
                args.uv_render_size,
                args.uv_sampling_mode,
                target_device,
            )
            with memory_monitor.track("Stage: UV generation"):
                mesh = generate_uv_ready_mesh(
                    mesh=mesh,
                    texture_size=args.uv_texture_size,
                    render_size=args.uv_render_size,
                    sampling_mode=args.uv_sampling_mode,
                    uv_device=target_device,
                )
            log_mesh_stats(mesh, "Mesh after UV mapping")

        with memory_monitor.track("Stage: export"):
            export_mesh(mesh, output_path, args.output_format)
        logger.info("Mesh exported to %s", output_path)
    finally:
        memory_monitor.report()
