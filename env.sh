#  python==3.10.14 cuda toolkit 12.4
# don't use python==3.10

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install "git+http://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
pip install pyassimp==5.2.5

#  pip install --upgrade numba==0.58.1 llvmlite==0.41.1
# pip uninstall onnxruntime rembg
# pip install onnxruntime==1.23.2  rembg==2.0.67
pip install "onnxruntime-gpu==1.20.0"
pip install "rembg[gpu]" # for library
# pip install "rembg[gpu,cli]" # for library + cli

cd step1x3d_texture/custom_rasterizer
python setup.py install
cd ../differentiable_renderer
python setup.py install
cd ../../