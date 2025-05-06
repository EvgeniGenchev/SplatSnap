poetry run splatsnap \
  --input-path ./data/crime.splat \
  --output-path output_image.jpg \
  --camera-position 0 0 5 \
  --look-at 0 0 0 \
  --up 0 -1 0 \
  --fov-deg 70 \
  --image-width 800 \
  --image-height 600 \
  --show-image
