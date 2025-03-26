import os
import json
import csv
import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils import load_image, load_vlm, get_image_list
import colorsys
import math
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info

def internvl25_inference(model, tokenizer, image_paths):
    questions = ['<image>\nWhich country has the largest shares?'] * len(image_paths)

    pixel_values = [load_image(image_path, max_num=12).to(torch.bfloat16).cuda() for image_path in image_paths]
    num_patches_list = [pixel_value.size(0) for pixel_value in pixel_values]
    pixel_values = torch.cat(pixel_values, dim=0)
        
    generation_config = dict(max_new_tokens=128, do_sample=False, return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.pad_token_id)
    response, scores = model.batch_chat(model,
                                        tokenizer,
                                        pixel_values,
                                        num_patches_list=num_patches_list,
                                        questions=questions,
                                        generation_config=generation_config)
    return response, scores

def qwen2_inference(model, processor, image_paths):
    questions = ['Which country has the largest shares?'] * len(image_paths)
    
    messages = []
    for question, image_path in zip(questions, image_paths):
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path
                    
                    },
                    {"type": "text", "text": question}
                ]
            }
        ]
        messages.append(message)
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt")        
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, 
                                   max_new_tokens=128,
                                   do_sample=False,
                                   return_dict_in_generate=True,
                                   output_scores=True)
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    scores = generated_ids.scores[0]
    return response, scores

def json2plot(json_file):
    with open(json_file, "r") as f:
        json_objects = json.load(f)
    
    probabilities = [object['probability'] for object in json_objects]
    xs = np.linspace(0, 1, len(probabilities))

    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 18})
    plt.plot(xs, probabilities, color='black')
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0, 0.5, 1.0])
    plt.xlabel('Contrast')
    plt.ylabel('Score')
    plt.tight_layout()

    plt.savefig(json_file.replace(".json", ".png"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="OpenGVLab/InternVL2_5-4B")
    parser.add_argument("--save_file", type=str, default="outputs/sensitivity/chart-color/InternVL2_5-4B")
    parser.add_argument("--image_dir", type=str, default="dataset/sensitivity/chart-color")
    parser.add_argument("--bs", type=int, default=100)
    args = parser.parse_args()
    # OpenGVLab/InternVL2_5-8B
    # Qwen/Qwen2-VL-2B-Instruct

    model, tokenizer = load_vlm(args.model_id)
    image_list = get_image_list(args.image_dir)
    
    json_objects = []
    for i in tqdm(range(0, len(image_list), args.bs), total=len(image_list)//args.bs):
        image_paths = image_list[i:i+args.bs]
        if "InternVL2_5" in args.model_id:
            responses, scores = internvl25_inference(model, tokenizer, image_paths)
        elif "Qwen2-VL" in args.model_id:
            responses, scores = qwen2_inference(model, tokenizer, image_paths)
        else:
            raise NotImplementedError
        
        for j, image_path in enumerate(image_paths):
            score = 1.0 if 'greenland' in responses[j].lower() else 0.0
            json_object = {}
            json_object["image_path"] = image_path
            json_object["probability"] = score
            json_object["response"] = responses[j]
            json_objects.append(json_object)

    with open(args.save_file, "w") as f:
        json.dump(json_objects, f)

    json2plot(args.save_file)