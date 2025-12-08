from SigLIP import SigLIPModel
import LOGS

import os
import sys
from PIL import Image
from time import perf_counter
import torch
import torchvision.transforms.functional as TF

IMAGE_FOLDER = "IMAGES"
OUTPUT_FOLDER = "OUTPUT"
IMAGE_SCALE = 512  # Resize images to have this width while maintaining aspect ratio
USE_GPU_RESIZE = True  # set True to try GPU resize (needs torchvision and CUDA)
THREAD_COUNT = 4  # number of threads for CPU processing


if __name__ == "__main__":
    try:
        user_search = sys.argv[1]
    except IndexError:
        LOGS.log_error("No search term provided. Please provide a search term as a command-line argument.")
        LOGS.log_info("Usage: python main.py <search_term>")
        sys.exit(1)

    LOGS.log_info("Initializing SigLIP model...")
    try:
        SigLip = SigLIPModel(use_compile=False)
        LOGS.log_success("SigLIP model initialized.")
    except Exception as e:
        LOGS.log_error(f"Failed to initialize SigLIP model: {e}")
        sys.exit(1)

    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        LOGS.log_success(f"Output folder '{OUTPUT_FOLDER}' is ready.")
    except Exception as e:
        LOGS.log_error(f"Failed to create output folder '{OUTPUT_FOLDER}': {e}")
        sys.exit(1)

    
    LOGS.log_info(f"Processing images in folder: {IMAGE_FOLDER} for search term: '{user_search}'")
    for image_file in os.listdir(IMAGE_FOLDER):
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        try:
            start_time = perf_counter()
            image = Image.open(image_path).convert("RGB")
            new_w = IMAGE_SCALE
            new_h = image.height * IMAGE_SCALE // image.width

            if USE_GPU_RESIZE and SigLip.use_cuda:
                tensor = TF.pil_to_tensor(image).to(SigLip.device, non_blocking=True)
                tensor = TF.resize(tensor, [new_h, new_w], antialias=True)
                image = TF.to_pil_image(tensor.cpu())
            else:
                image = image.resize((new_w, new_h))

            output = SigLip.forward(image=image, texts=[user_search])
            end_time = perf_counter()
            LOGS.log_info(f"Processing time for image {image_file}: {end_time - start_time:.4f} seconds")
            for i, (text, confidence) in enumerate(output):
                LOGS.log_info(f"Image: {image_file}, Confidence for '{user_search}': {confidence:.2%}")
                if confidence > 0.3:
                    output_path = os.path.join(OUTPUT_FOLDER, image_file)
                    image.save(output_path)
                    LOGS.log_success(f"Image '{image_file}' saved to '{OUTPUT_FOLDER}' with confidence {confidence:.2%}")

        except Exception as e:
            LOGS.log_error(f"Failed to process image {image_file}: {e}")