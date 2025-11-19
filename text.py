import gc
import torch
torch.cuda.set_per_process_memory_fraction(0.95)
# Stage 1: 3D geometry generation
# from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline

# # define the pipeline
# geometry_pipeline = Step1X3DGeometryPipeline.from_pretrained(
#     "stepfun-ai/Step1X-3D",
#     subfolder="Step1X-3D-Geometry-1300m",
#     # torch_dtype=torch.float16,
# )
# # geometry_pipeline.enable_model_cpu_offload()

# # input image
# input_image_path = "examples/images/000.png"

# # run pipeline and obtain the untextured mesh 
# generator = torch.Generator(device=geometry_pipeline.device).manual_seed(2025)
# out = geometry_pipeline(input_image_path, guidance_scale=7.5, num_inference_steps=50)

# # export untextured mesh as .glb format
# out.mesh[0].export("untexture_mesh.glb")
# # free geometry pipeline weights before loading the texture stage
# geometry_pipeline = None
# torch.cuda.empty_cache()
# gc.collect()


# Stage 2: 3D texure synthsis
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
    Step1X3DTexturePipeline,
)
import trimesh

# load untextured mesh
untexture_mesh = trimesh.load("output/captured.glb")

# define texture_pipeline
texture_pipeline = Step1X3DTexturePipeline.from_pretrained(
    "stepfun-ai/Step1X-3D",
    subfolder="Step1X-3D-Texture",
    render_size=1024,
    texture_size=1024,
    mv_image_size=640,
    # enable_cpu_offload=True,
    # dtype=torch.float16,
)

# reduce face
# untexture_mesh = remove_degenerate_face("output/captured.glb")

# texture mapping
textured_mesh = texture_pipeline("/home/farsee2/project/objgen/examples/imgs/captured.jpeg", untexture_mesh, remove_bg=False)

# export textured mesh as .glb format
textured_mesh.export("textured_mesh.glb")
