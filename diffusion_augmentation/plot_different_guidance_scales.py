import os

import matplotlib.pyplot as plt
import torch

from diffusers import DiffusionPipeline


save_dir = "emg_images_generated-from-diffusion/"
pipeline = DiffusionPipeline.from_pretrained("examples/dreambooth/path_to_saved_model_subject0_emg", torch_dptype=torch.float16, use_safetensors=True).to("cuda")

guidance_scales = range(32, 64)
images = []

for scale in guidance_scales:
    image = pipeline("tnu heatmap from subject 0", num_inference_steps=50, guidance_scale=scale).images[0]
    images.append(image)

fig, axes = plt.subplots(8, 4, figsize=(12, 24))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(images[i])
    ax.axis("off")
    ax.set_title(f"Guidance Scale: {i}")

plt.tight_layout()

os.makedirs(save_dir, exist_ok=True)

# Save the image
plt.savefig(f"{save_dir}guidance_scales_2.png")

# Show the image
plt.show()
