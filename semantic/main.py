import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils import load_image, load_vlm
from PIL import Image
import matplotlib.pyplot as plt

@torch.no_grad()
def clip_inference(model, tokenizer, ref_color, target_colors):
    ref_pixel_value = load_image(ref_color, model_id="clip")
    target_pixel_values = [load_image(target_color, model_id="clip") for target_color in target_colors]

    ref_pixel_value = tokenizer(images=ref_pixel_value, return_tensors="pt")
    target_pixel_values = tokenizer(images=target_pixel_values, return_tensors="pt")

    ref_pixel_value = {k: v.cuda() for k, v in ref_pixel_value.items()}
    target_pixel_values = {k: v.cuda() for k, v in target_pixel_values.items()}

    ref_feature = model.get_image_features(**ref_pixel_value)
    target_features = model.get_image_features(**target_pixel_values)
    scores = torch.nn.functional.cosine_similarity(ref_feature, target_features, dim=1)
    return scores

def resize_shape(example, size=1536):
    example = Image.fromarray(example)
    example = example.resize((size, size))
    example = np.asarray(example).copy()
    return example

def center_crop(example):
    h, w = example.shape[:2]

    center_h, center_w = h // 2, w // 2
    size = min(h, w) // 2

    example = example[center_h - size: center_h + size, center_w - size: center_w + size]
    return example

def extract_patch(image, kernel_size, stride):
    h, w = image.shape[:2]
    patches = []
    for i in range(0, h - kernel_size, stride):
        for j in range(0, w - kernel_size, stride):
            patches.append(image[i: i + kernel_size, j: j + kernel_size])
    return patches

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--sample1", type=str, default="./semantic/target.JPEG")
    parser.add_argument("--sample2", type=str, default="./semantic/2.JPEG")
    parser.add_argument("--bs", type=int, default=1000)
    args = parser.parse_args()

    model, tokenizer = load_vlm(args.model_id)

    image1, image2 = Image.open(args.sample1).convert("RGB"), Image.open(args.sample2).convert("RGB")
    image1, image2 = np.asarray(image1).copy(), np.asarray(image2).copy()
    image1, image2 = resize_shape(center_crop(image1)), resize_shape(center_crop(image2))   
    kernel_sizes = [260, 240, 220, 200, 180, 160, 140, 120, 100, 80, 60]
    stride = 40

    for kernel_size in kernel_sizes:
        patches = extract_patch(image2, kernel_size, stride)
        score_list = []
        for i in tqdm(range(0, len(patches), args.bs), total=len(patches)//args.bs):
            patch_batch = patches[i:i+args.bs]
            patch_batch = np.array(patch_batch)
            scores = clip_inference(model, tokenizer, image1, patch_batch)
            score_list.append(scores.cpu())
        score_list = torch.cat(score_list, dim=0)
        sqrt_size = int(np.sqrt(len(patches)))
        score_map = score_list.reshape(sqrt_size, sqrt_size).view(1, 1, sqrt_size, sqrt_size)
        score_map = torch.nn.functional.interpolate(score_map, size=(256, 256), mode='bilinear', align_corners=False)
        score_map = score_map.squeeze().cpu().numpy()
        sizes = np.shape(score_map)
        fig = plt.figure()
        fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(score_map, cmap='plasma')
        plt.savefig(f"./semantic/{kernel_size}.jpg", dpi = sizes[0])
        plt.close()
