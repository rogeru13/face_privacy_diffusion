# face_privacy_diffusion
- Use Flux1B as off-the-shelf inpainting model
- Apply facial recognition algorithms to detect faces in Laion-400M dataset (Start with 1% or 0.1% sample of LAION dataset for initial testing)
- Replace detected faces with AI-generated faces using Flux


## clipstable_diff: face privacy diffusion model
 - *Though clipstable_diff.ipynb cannot be rendered in GitHub, attempt opening it in Google Colab to view test images and outputs*
 - Originally trained on ~2,000 Celeb-A PNG images + .txt files detailing yaw, pitch, roll and edge-coordinates of the face
 - Used RunwayML as base inpainting mode
 - Finetuning conducted using LoRA Integration & CLIP based reward system, targeting and rewarding "a realistic, detailed human face"
 - During anonymization process, MTCNN model detects face and landmarks, superimposing a default face mesh, then removing the detected facial area
 - Fine-tuned model inpaints on the face mesh resulting in higher inpainting output quality
