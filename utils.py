import os
import requests
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPImageProcessor
from qwen_vl_utils import process_vision_info
from transformers.image_utils import load_image as hf_load_image
from generation_utils import internvl_2_5_chat, internvl_2_5_batch_chat
from typing import Union


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file,
               input_size=448,
               max_num=12,
               model_id="OpenGVLab/InternVL2_5-1B"):
    if type(image_file) == str and image_file.startswith("http"):
        image = Image.open(requests.get(image_file, stream=True).raw).convert("RGB")
    elif type(image_file) == str:
        image = Image.open(image_file).convert('RGB')
    elif type(image_file) == np.ndarray:
        image = Image.fromarray(image_file.astype(np.uint8)).convert('RGB')
    else:
        raise ValueError("Invalid image type")
    
    if "InternVL2_5" in model_id:
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    else:
        return image

def get_image_list(image_dir: str) -> list[str]:
    image_list = os.listdir(image_dir)
    image_list = [img for img in image_list if img.endswith(".jpg")]
    def myFunc(e):
        return int(e.split(".")[0])
    image_list.sort(key=myFunc)
    image_list = [os.path.join(image_dir, img) for img in image_list]
    return image_list

def load_vlm(model_id: str) -> tuple[Union[AutoModel, AutoModelForVision2Seq, Qwen2VLForConditionalGeneration, CLIPModel], Union[AutoTokenizer, AutoProcessor, CLIPProcessor]]:
    if "InternVL2_5" in model_id:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True, 
            trust_remote_code=True).eval().cuda()
        model.chat = internvl_2_5_chat
        model.batch_chat = internvl_2_5_batch_chat

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        return model, tokenizer
    elif "SmolVLM" in model_id:
        tokenizer = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2"
        )
        model = model.eval().cuda()

        return model, tokenizer
    elif "Qwen2-VL" in model_id:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

        return model, processor
    elif "clip" in model_id or "CLIP" in model_id:
        model = CLIPModel.from_pretrained(model_id)
        model = model.eval().cuda()

        processor = CLIPProcessor.from_pretrained(model_id)
        return model, processor
    elif "InternViT" in model_id:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()

        image_processor = CLIPImageProcessor.from_pretrained(model_id)
        return model, image_processor
    elif "siglip" in model_id:
        model = AutoModel.from_pretrained(model_id).eval().cuda()
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor

if __name__ == "__main__":
    from transformers import AutoProcessor, AutoModel
    model_id = "google/siglip-so400m-patch14-384"

    model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, padding="max_length", return_tensors="pt")
    import pdb; pdb.set_trace()


    