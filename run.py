from PIL import Image

from fabric.generator import AttentionBasedGenerator
from fabric.iterative import IterativeFeedbackGenerator
from fabric.utils import get_free_gpu, tile_images
import torch
import os
from pathlib import Path

def get_feedback(images) -> tuple[list[Image.Image], list[Image.Image]]:
    liked, disliked = [], []
    for image in images:
        image.show()
        answer = input("Enter yes or no: ")
        if answer == "yes":
            liked.append(image)
        elif answer == "no":
            disliked.append(image)
    return liked, disliked

base_generator = AttentionBasedGenerator("dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16)
base_generator.to("cuda")

generator = IterativeFeedbackGenerator(base_generator)

prompt = "photo of a big house at Cornell, masterpiece, best quality, fine details"
negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

path = './cornell/'
Path(path).mkdir(exist_ok=True)

for _ in range(10):
    images: list[Image.Image] = generator.generate(prompt, negative_prompt=negative_prompt)

    tiled = tile_images(images)
    tiled_path = os.path.join(path, f"tiled{_}.png")
    tiled.save(tiled_path)
    liked, disliked = get_feedback(images)
    generator.give_feedback(liked, disliked)
generator.reset()
