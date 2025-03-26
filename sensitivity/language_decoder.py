
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

from PIL import Image
import base64
from io import BytesIO

@torch.no_grad()
def internvl25_inference(model, tokenizer, red_color, target_colors):
    questions = ['<image>\nIn one word, describe the image:'] * (len(target_colors) + 1)
    ref_pixel_value = load_image(red_color, max_num=12).to(torch.bfloat16).cuda()
    target_pixel_values = [load_image(target_color, max_num=12).to(torch.bfloat16).cuda() for target_color in target_colors]
    
    pixel_values = [ref_pixel_value] + target_pixel_values
    num_patches_list = [ref_pixel_value.size(0)] + [target_pixel_value.size(0) for target_pixel_value in target_pixel_values]
    pixel_values = torch.cat(pixel_values, dim=0)
    
    generation_config = dict(max_new_tokens=1,
                             do_sample=False,
                             return_dict_in_generate=True,
                             output_scores=True,
                             pad_token_id=tokenizer.pad_token_id,
                             output_hidden_states=True)
    response, _, generation_output = model.batch_chat(model,
                                                      tokenizer,
                                                      pixel_values,
                                                      num_patches_list=num_patches_list,
                                                      questions=questions,
                                                      generation_config=generation_config)
    last_hidden_states = generation_output.hidden_states[0][-1][:, -1, :]

    ref_feature = last_hidden_states[0].unsqueeze(0)
    target_features = last_hidden_states[1:]

    scores = torch.nn.functional.cosine_similarity(ref_feature, target_features, dim=1)
    return scores

def numpy2base64(image):
    image = Image.fromarray(image.astype(np.uint8))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@torch.no_grad()
def qwen2_inference(model, processor, ref_color, target_colors):
    # convert ref_color (numpy, uint8) to Base64 encoded image
    ref_color = numpy2base64(ref_color)
    target_colors = [numpy2base64(target_color) for target_color in target_colors]
    questions = ['In one word, describe the image:'] * (len(target_colors))
    
    messages = []
    message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "data:image;base64,"+ref_color
                    
                    },
                    {"type": "text", "text": questions[0]}
                ]
            }
        ]
    messages.append(message)
    for question, image_path in zip(questions, target_colors):
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "data:image;base64,"+image_path
                    
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
                                   max_new_tokens=1,
                                   do_sample=False,
                                   return_dict_in_generate=True,
                                   output_scores=True,
                                   output_hidden_states=True)
    
    ref_feature = generated_ids['hidden_states'][0][-1][0, -1].unsqueeze(0)
    target_features = generated_ids['hidden_states'][0][-1][1:, -1]
    scores = torch.nn.functional.cosine_similarity(ref_feature, target_features, dim=1)
    return scores

def color_json2plot(json_file):
    rho = np.linspace(0, 1, 100)
    phi = np.linspace(0, math.pi * 2, 500)
    RHO, PHI = np.meshgrid(rho, phi)
    h = (PHI - PHI.min()) / (PHI.max() - PHI.min())
    h = np.flip(h)
    s = RHO
    v = np.ones_like(h)
    h, s, v = h.flatten().tolist(), s.flatten().tolist(), v.flatten().tolist()
    target_colors = [colorsys.hsv_to_rgb(*x) for x in zip(h, s, v)]
    target_colors = np.array(target_colors)

    with open(json_file, "r") as f:
        json_objects = json.load(f)
    
    probabilities = [object['probability'] for object in json_objects]
    assert len(probabilities) == len(target_colors)
    probabilities = np.array(probabilities).reshape(500, 100)
    # min-max normalization
    # probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())

    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.pcolormesh(PHI, RHO, probabilities, cmap='plasma', vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(json_file.replace(".json", ".png"))
    plt.close()

    dr = rho[1] - rho[0]
    dphi = phi[1] - phi[0]
    sac = 0.0
    for r, p, prob in zip(RHO.flatten(), PHI.flatten(), probabilities.flatten()):
        sac += r * dr * dphi * prob
    
    with open(json_file.replace(".json", ".txt"), "w") as f:
        f.write(f"Sensitivity Area of Color (SAC): {sac}")
    print(f"Sensitivity Area of Color (SAC): {sac}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    # OpenGVLab/InternVL2_5-2B
    # Qwen/Qwen2-VL-2B-Instruct
    parser.add_argument("--save_file", type=str, default="outputs/sensitivity/color_language_decoder/Qwen2-VL-7B-Instruct.json")
    parser.add_argument("--color", type=str, default="red")
    parser.add_argument("--bs", type=int, default=10)
    args = parser.parse_args()

    if args.color == "red":
        ref_color = (255, 0, 0)
    elif args.color == "green":
        ref_color = (0, 255, 0)
    elif args.color == "blue":
        ref_color = (0, 0, 255)
    

    rho = np.linspace(0, 1, 100)
    phi = np.linspace(0, math.pi * 2, 500)
    RHO, PHI = np.meshgrid(rho, phi)
    h = (PHI-PHI.min()) / (PHI.max()-PHI.min()) # use angle to determine hue, normalized from 0-1
    h = np.flip(h)
    s = RHO               # saturation is set as a function of radias
    v = np.ones_like(RHO) # value is constant
    h,s,v = h.flatten().tolist(), s.flatten().tolist(), v.flatten().tolist()
    target_colors = [colorsys.hsv_to_rgb(*x) for x in zip(h,s,v)]
    target_colors = np.array(target_colors)
    target_colors = (target_colors * 255).astype(np.uint8)

    model, tokenizer = load_vlm(args.model_id)
    json_objects = []
    count = 0
    for i in tqdm(range(0, len(target_colors), args.bs), total=len(target_colors)//args.bs):
        # ref_color and target_color to image
        ref_color_img = np.ones((224, 224, 3), dtype=np.uint8) * ref_color
        bs = min(args.bs, len(target_colors[i:i+args.bs]))
        target_color_imgs = np.ones((bs, 224, 224, 3), dtype=np.uint8) * target_colors[i:i+args.bs].reshape(-1, 1, 1, 3)
        
        if "InternVL2_5" in args.model_id:
            scores = internvl25_inference(model, tokenizer, ref_color_img, target_color_imgs)
        elif "Qwen2" in args.model_id:
            scores = qwen2_inference(model, tokenizer, ref_color_img, target_color_imgs)

        for j, target_color in enumerate(target_colors[i:i+args.bs]):
            json_object = {}
            json_object["probability"] = scores[j].item()
            json_objects.append(json_object)

    save_file = args.save_file.replace(".json", f"_{args.color}.json")
    with open(save_file, "w") as f:
        json.dump(json_objects, f)
    
    color_json2plot(save_file)