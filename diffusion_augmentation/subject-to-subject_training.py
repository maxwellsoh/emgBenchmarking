from dataclasses import dataclass
import argparse
from datasets import load_dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from diffusers import UNet2DModel
import torch
from PIL import Image
from diffusers import DDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
from accelerate import Accelerator, notebook_launcher
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import glob


args = argparse.ArgumentParser()
args.add_argument("--image_size", type=int, default=224)
args.add_argument("--dataset", type=str, default="OzdemirEMG")
args.add_argument("--preprocessing_method", type=str, default="cwt")
args.add_argument("--left_out_subject", type=int, default=1)

args = args.parse_args()

high_performing_subjects = [8, 9, 19, 20, 27]

@dataclass
class TrainingConfig:
    image_size = args.image_size  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 8  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 3
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = f"diffusion_augmentation/custom_models/{args.dataset}/{args.preprocessing_method}/subject-to-subject/{args.left_out_subject}"  # the model name locally and on the HF Hub

    seed = 0

config = TrainingConfig()
config.dataset_name = args.dataset # will need to change parent folders to allow for "subject to subject" dataset
dataset = load_dataset(config.dataset_name, split="train")

SHOW_TRAINING_IMAGES = False
if SHOW_TRAINING_IMAGES:
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i, image in enumerate(dataset[:4]["input_image"]):
        axs[i].imshow(image)
        axs[i].set_axis_off()
    fig.show()

preprocess = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def transform(examples):
    # Assume `examples` has 'input_image' and 'target_image' fields
    inputs = [preprocess(image.convert("RGB")) for image in examples["input_image"]]
    targets = [preprocess(image.convert("RGB")) for image in examples["target_image"]]
    return {"input_image": inputs, "target_image": targets}

dataset.set_transform(transform)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512), 
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    ),
)

sample_image = dataset[0]["input_image"].unsqueeze(0)
print("Input shape:", sample_image.shape)

print("Output shape:", model(sample_image, timestep=0).sample.shape)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

Image.fromarray(((noisy_image.permute(0,2,3,1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])
    
noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps = config.lr_warmup_steps,
    num_training_steps = (len(train_dataloader) * config.num_epochs)
)

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=2)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            # clean_images = batch["image"]
            input_images = batch["input_image"]
            target_images = batch["target_image"]
            # Sample noise to add to the images
            noise = torch.randn(input_images.shape, device=input_images.device)
            bs = input_images.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=input_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(input_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                predicted_target_image = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(target_images, predicted_target_image)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)
                    
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])