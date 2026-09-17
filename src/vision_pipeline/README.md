## Enrolamiento de identidad (face_id_node)

Este repositorio no incluye `mario_embedding.npy` ni `mi_foto.jpg` por privacidad. Antes de correr `face_id_node`, genera tu propio embedding:

\`\`\`bash
cd src/vision_pipeline/vision_pipeline
python3 embedding_generator.py
\`\`\`

El script toma una foto desde la cámara y guarda el embedding correspondiente en `mario_embedding.npy`, que `face_id_node` carga al arrancar.