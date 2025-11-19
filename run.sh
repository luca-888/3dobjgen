python main.py \
  --input examples/imgs/captured.jpeg \
  --output-dir output \
  --output-name captured \
  --output-format obj \
  --auto-uv \
  --uv-texture-size 1024 \
  --uv-render-size 512 \
  --uv-sampling-mode nearest \
  --uv-device cuda \
  --rembg-backend u2net \
  # --disable-rembg-cache 
  # --rembg-backend u2net \
