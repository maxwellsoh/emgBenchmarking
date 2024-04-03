from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load your pipeline
pipeline = DiffusionPipeline.from_pretrained("examples/text_to_image/custom_models/emg-loso-model_subject-1", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

# Define gestures and guidance scales
gestures = ["Rest", "Flexion", "Abduction", "Extension", "Grip", "Radial_Deviation", "Ulnar_Deviation"]
guidance_scales = [5, 15, 25]

for scale in guidance_scales:
    # Set up the figure for plotting
    fig, axs = plt.subplots(5, 7, figsize=(20, 15))
    fig.suptitle(f'Guidance Scale {scale}', fontsize=16)

    for j, gesture in enumerate(gestures):
        for i in range(5):
            # Generate the image
            image = pipeline(f"tnu {gesture} heatmap for loso-cv subject 1", num_inference_steps=50, guidance_scale=scale).images[0]
            image = image.resize((224,224))

            # Plot the image
            axs[i, j].imshow(np.asarray(image))
            axs[i, j].axis('off')
            axs[i, j].set_title(gesture)

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"guidance_scale_{scale}.png")
    plt.close()
