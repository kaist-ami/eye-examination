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
    if 'size' in image_paths[0]:
        questions = ['<image>\nAre the two samples identical in size? Answer with yes or no.'] * len(image_paths)
    elif 'shape' in image_paths[0]:
        questions = ['<image>\nAre the two samples identical in shape? Answer with yes or no.'] * len(image_paths)
    elif 'color' in image_paths[0]:
        questions = ['<image>\nAre the color boxes the same color? Answer with yes or no.'] * len(image_paths)

    pixel_values = [load_image(image_path, max_num=12).to(torch.bfloat16).cuda() for image_path in image_paths]
    num_patches_list = [pixel_value.size(0) for pixel_value in pixel_values]
    pixel_values = torch.cat(pixel_values, dim=0)
        
    generation_config = dict(max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.pad_token_id)
    response, scores = model.batch_chat(model,
                                        tokenizer,
                                        pixel_values,
                                        num_patches_list=num_patches_list,
                                        questions=questions,
                                        generation_config=generation_config)
    return response, scores

def qwen2_inference(model, processor, image_paths):
    if 'size' in image_paths[0]:
        questions = ['Are the two samples identical in size? Answer with yes or no.'] * len(image_paths)
    elif 'shape' in image_paths[0]:
        questions = ['Are the two samples identical in shape? Answer with yes or no.'] * len(image_paths)
    elif 'color' in image_paths[0]:
        questions = ['Are the color boxes the same color? Answer with yes or no.'] * len(image_paths)
    
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
                                   max_new_tokens=1,
                                   do_sample=False,
                                   return_dict_in_generate=True,
                                   output_scores=True)
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    scores = generated_ids.scores[0]
    return response, scores

def smolvlm_inference(model, processor, image_paths):
    if 'size' in image_paths[0]:
        questions = ['Are the two samples identical in size? Answer with yes or no.'] * len(image_paths)
    elif 'shape' in image_paths[0]:
        questions = ['Are the two samples identical in shape? Answer with yes or no.'] * len(image_paths)
    elif 'color' in image_paths[0]:
        questions = ['Are the color boxes the same color? Answer with yes or no.'] * len(image_paths)
    images = [[load_image(image_path, model_id="SmolVLM")] for image_path in image_paths]
    
    prompts = []
    for question in questions:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        prompts.append(prompt)
    inputs = processor(text=prompts, images=images, return_tensors="pt").to('cuda')
    generated_ids = model.generate(**inputs, 
                                   max_new_tokens=1,
                                   do_sample=False,
                                   return_dict_in_generate=True,
                                   output_scores=True)
    sequences = generated_ids.sequences
    responses = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    responses = [response.split('Assistant: ')[-1] for response in responses]
    scores = generated_ids.scores[0]
    return responses, scores

def eccentricity_json2csv(json_file):
    csv_file = json_file.replace(".json", ".csv")

    with open(json_file, "r") as f:
        json_objects = json.load(f)
    
    rows = np.linspace(0, 1.0, 1000)
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['score'])
        for row, object in zip(rows, json_objects):
            probability = float(object['probability'])
            writer.writerow([row, probability])
    
    print(f"Saved to {csv_file}")
    rows = np.linspace(0, 1.0, 1000)[:900]
    
    d = rows[1] - rows[0]
    sas = 0
    for i in range(len(rows)):
        object = json_objects[i]
        probability = float(object['probability'])
        sas = sas + d * probability

    txt_file = json_file.replace(".json", ".txt")
    with open(txt_file, "w") as f:
        f.write(f"Sensitivity Area of Shape (SAS): {sas}")
    print(f"Saved to {txt_file}")
    print(f"Sensitivity Area of Shape (SAS): {sas}")

def poly_json2csv(json_file):
    csv_file = json_file.replace(".json", ".csv")

    with open(json_file, "r") as f:
        json_objects = json.load(f)
    
    rows = np.arange(4, 361, 1)
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['score'])
        for row, object in zip(rows, json_objects):
            probability = float(object['probability'])
            writer.writerow([row, probability])
    
    print(f"Saved to {csv_file}")
    rows = np.arange(4, 361, 1)[:27]
    
    d = rows[1] - rows[0]
    sas = 0
    for i in range(len(rows)):
        object = json_objects[i]
        probability = float(object['probability'])
        sas = sas + d * probability

    txt_file = json_file.replace(".json", ".txt")
    with open(txt_file, "w") as f:
        f.write(f"Sensitivity Area of Shape (SAS): {sas}")
    print(f"Saved to {txt_file}")
    print(f"Sensitivity Area of Shape (SAS): {sas}")

def size_json2csv(json_file):
    csv_file = json_file.replace(".json", ".csv")

    with open(json_file, "r") as f:
        json_objects = json.load(f)
    
    rows = np.arange(0, 200, 1)
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['score'])
        for row, object in zip(rows, json_objects):
            probability = float(object['probability'])
            writer.writerow([row, probability])
    
    print(f"Saved to {csv_file}")
    rows = np.arange(0, 200, 1)
    
    d = rows[1] - rows[0]
    sas = 0
    for i in range(len(rows)):
        object = json_objects[i]
        probability = float(object['probability'])
        sas = sas + d * probability

    txt_file = json_file.replace(".json", ".txt")
    with open(txt_file, "w") as f:
        f.write(f"Sensitivity Area of Shape (SAS): {sas}")
    print(f"Saved to {txt_file}")
    print(f"Sensitivity Area of Shape (SAS): {sas}")

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
    # probabilities = [probabilities > 0.5 for probabilities in probabilities]
    assert len(probabilities) == len(target_colors)
    probabilities = np.array(probabilities).reshape(500, 100)

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
    parser.add_argument("--model_id", type=str, default="OpenGVLab/InternVL2_5-8B")
    # OpenGVLab/InternVL2_5-1B
    # OpenGVLab/InternVL2_5-4B
    # OpenGVLab/InternVL2_5-8B
    parser.add_argument("--save_file", type=str, default="outputs/sensitivity/InternVL2_5-8B_eccentricity.json")
    parser.add_argument("--image_dir", type=str, default="dataset/sensitivity/eccentricity/shape-eccentricity")
    parser.add_argument("--bs", type=int, default=10)
    args = parser.parse_args()

    # model, tokenizer = load_vlm(args.model_id)
    # image_list = get_image_list(args.image_dir)

    # yes_texts, no_texts = ["yes", "Yes", "YES"], ["no", "No", "NO"]
    # if "InternVL2_5" in args.model_id:
    #     yes_index, no_index = [tokenizer.convert_tokens_to_ids(token) for token in yes_texts], [tokenizer.convert_tokens_to_ids(token) for token in no_texts]
    # elif "SmolVLM" in args.model_id or "Qwen2-VL" in args.model_id:
    #     yes_index, no_index = [tokenizer.tokenizer.convert_tokens_to_ids(token) for token in yes_texts], [tokenizer.tokenizer.convert_tokens_to_ids(token) for token in no_texts]
    
    # json_objects = []
    # for i in tqdm(range(0, len(image_list), args.bs), total=len(image_list)//args.bs):
    #     image_paths = image_list[i:i+args.bs]
    #     if "InternVL2_5" in args.model_id:
    #         responses, scores = internvl25_inference(model, tokenizer, image_paths)
    #     elif "SmolVLM" in args.model_id:
    #         responses, scores = smolvlm_inference(model, tokenizer, image_paths)
    #     elif "Qwen2-VL" in args.model_id:
    #         responses, scores = qwen2_inference(model, tokenizer, image_paths)
    #     else:
    #         raise NotImplementedError

    #     for j, image_path in enumerate(image_paths):
    #         yes_score, no_score = scores[j][yes_index], scores[j][no_index]
    #         all_scores = torch.cat([yes_score, no_score])
    #         probability = torch.softmax(all_scores, dim=0)
    #         probability = (probability[0] + probability[1] + probability[2]).item()

    #         json_object = {}
    #         json_object["image_path"] = image_path
    #         json_object["probability"] = probability
    #         json_objects.append(json_object)

    # with open(args.save_file, "w") as f:
    #     json.dump(json_objects, f)
    
    if "eccentricity" in args.save_file:
        eccentricity_json2csv(args.save_file)
    elif "poly" in args.save_file:
        poly_json2csv(args.save_file)
    elif "size" in args.save_file:
        size_json2csv(args.save_file)
    elif "color" in args.save_file:
        color_json2plot(args.save_file)