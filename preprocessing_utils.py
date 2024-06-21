import torch
from PIL import Image

def denormalize_imagenet(tensor_to_denormalize):
    """
    De-normalizes an image tensor that has been normalized using ImageNet normalization parameters.

    Args:
        tensor_to_denormalize (torch.Tensor): The image tensor to be de-normalized.

    Returns:
        torch.Tensor: The de-normalized image tensor.
    """
    
    # Standard ImageNet normalization parameters
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # De-normalize the images
    tensor_denormalized = tensor_to_denormalize * imagenet_std + imagenet_mean
    
    return tensor_denormalized

def denormalized_tensor_to_pil_images(tensor_to_convert):
    """
    Converts a denormalized tensor to a list of PIL images.

    Args:
        tensor_to_convert (torch.Tensor): The tensor to convert. Should be denormalized.

    Returns:
        list: A list of PIL images.

    """
    
    # Ensure tensor is in the correct range
    tensor_byte = tensor_to_convert.clamp(0, 1) * 255

    # Convert tensor to numpy array and permute dimensions to (batch, height, width, channels)
    tensor_numpy = tensor_byte.permute(0, 2, 3, 1).byte().numpy()

    # Function to convert a batch of numpy arrays to a list of PIL images
    def batch_to_pil_images(np_images):
        pil_images = [Image.fromarray(np_images[i]) for i in range(np_images.shape[0])]
        return pil_images

    # Convert the batch to PIL images
    pil_images = batch_to_pil_images(tensor_numpy)

    return pil_images


