# vision_pipeline

## Modelo de detección facial

Los pesos de YOLOv8n-face no están incluidos en el repositorio (ver `.gitignore`). Descárgalos antes de correr `face_detector_node`:

\`\`\`bash
mkdir -p src/vision_pipeline/vision_pipeline/models
cd src/vision_pipeline/vision_pipeline/models
wget https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt
\`\`\`