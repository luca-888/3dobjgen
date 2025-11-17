# python 3.10 cuda toolkit 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu==1.15.1
pip install "rembg[gpu]" # for library
pip install "rembg[gpu,cli]" # for library + cli