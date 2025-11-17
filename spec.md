## Objective

Generate production-ready 3D objects with PBR materials from single images.

### Architecture

**Step 1: Background Removal**

- **Tool:** Rembg with BiRefNet model: https://github.com/danielgatis/rembg
- **Input:** Single RGB image
- **Output:** RGBA image with transparent background
- **Processing time:** <1 second

**Step 2: 3D Mesh Generation**

- **Primary:** Step1X-3D for quality (50-60 seconds total): https://github.com/stepfun-ai/Step1X-3D
- **Input:** RGBA image from Step 1, optional text caption
- **Output:** Untextured watertight mesh with UV mapping
- **Format:** GLB, OBJ, FBX

**Step 3: Mesh Refinement**

- **Tool:** meshoptimizer for decimation/optimization
- **Optional:** DetailGen3D for AI-based enhancement
- **Purpose:** Reduce polygon count while preserving silhouette
- **Target:** 5K-50K triangles depending on use case

**Step 4: PBR Texture Generation**

- **Tool:** MeshGen or DreamMat
- **Input:** Mesh from Step 3
- **Output:** Complete PBR material set
    - Albedo (base color)
    - Metallic map
    - Roughness map
    - Normal map
- **Resolution:** 2K-4K

**Step 5: Texture Application**

- **Tool:** Trimesh or Blender headless
- **Purpose:** Combine mesh + PBR textures into final asset
- **Output:** GLB with embedded materials

<aside>
💡

Output examples: 

[40571253_PS01_S01_NV01_RQP3_4.0_109cfa388c0e4c34ab0537f0b4dd288b.glb](attachment:9256f262-7af8-4619-8793-ed0d2240786e:40571253_PS01_S01_NV01_RQP3_4.0_109cfa388c0e4c34ab0537f0b4dd288b.glb)

[65ba07fa53d0d12ef79bda61453a53a0-G-79434401-a8fdeb7892e395432aa35178590c6b8d469d2da2-simple.glb](attachment:1fbbf420-b3b6-45d1-afd5-db137a011405:65ba07fa53d0d12ef79bda61453a53a0-G-79434401-a8fdeb7892e395432aa35178590c6b8d469d2da2-simple.glb)

</aside>

### Tech Stack

- Rembg (MIT)
- Step1X-3D (Apache 2.0) or InstantMesh (Apache 2.0)
- MeshGen (open-source, CVPR 2025) or DreamMat (SIGGRAPH 2024)
- meshoptimizer (MIT)
- Trimesh (MIT)
- Python 3.10+, PyTorch, CUDA

### Performance Targets

- **Quality mode:** 60-90 seconds per object
- **Speed mode:** 15-20 seconds per object

### Acceptance Criteria

**AC1: Background Removal**

- [ ]  Clean extraction with no visible background artifacts
- [ ]  Preserves fine details (hair, thin structures)
- [ ]  Alpha channel has smooth edges (no aliasing)

**AC2: Mesh Quality**

- [ ]  Watertight topology (no holes or non-manifold edges)
- [ ]  Proper UV mapping with minimal distortion
- [ ]  Clean quad-dominant topology suitable for subdivision
- [ ]  Geometry captures essential features from input image

**AC3: PBR Material Quality**

- [ ]  Albedo map free from baked-in lighting/shadows
- [ ]  Metallic map correctly identifies metal vs. non-metal
- [ ]  Roughness map shows appropriate surface variation
- [ ]  Normal map adds geometric detail without artifacts
- [ ]  Materials render correctly in Blender/Unity/Unreal

**AC4: Format Compatibility**

- [ ]  GLB exports successfully to web viewers (three.js, Babylon.js)
- [ ]  FBX imports cleanly into Unity and Unreal Engine
- [ ]  Materials automatically assign or require minimal manual adjustment
- [ ]  File size under 10MB for typical object

**AC5: Performance**

- [ ]  Quality pipeline completes in under 90 seconds on RTX 4090
- [ ]  Speed pipeline completes in under 20 seconds
- [ ]  Batch processing achieves 40+ objects/hour

**AC6: Visual Quality**

- [ ]  Output passes blind A/B test vs. manual modeling 70%+ of time
- [ ]  No visible seams in texture maps
- [ ]  Geometry silhouette matches input image within 5% deviation
- [ ]  Materials look physically plausible under standard lighting

## Non-Functional Requirements

**Reliability:**

- 95% success rate on well-captured inputs
- Graceful degradation on poor inputs (return error codes, not crashes)
- Automatic retry with fallback methods on failure

**Observability:**

- Log processing time for each pipeline stage
- Track GPU memory usage and detect OOM before crash
- Report quality metrics (dimensional accuracy, mesh validity)

**Scalability:**

- Support batch processing (multiple scenes/objects in queue)
- Horizontal scaling via multiple GPU workers
- Cloud deployment on AWS/Azure/GCP with GPU instances