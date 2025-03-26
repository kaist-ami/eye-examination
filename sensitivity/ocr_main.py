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
    questions = ['<image>\nWhat is the text in the image? Answer should be a single word.'] * len(image_paths)

    pixel_values = [load_image(image_path, max_num=12).to(torch.bfloat16).cuda() for image_path in image_paths]
    num_patches_list = [pixel_value.size(0) for pixel_value in pixel_values]
    pixel_values = torch.cat(pixel_values, dim=0)
        
    generation_config = dict(max_new_tokens=32, do_sample=False, return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.pad_token_id)
    response, scores = model.batch_chat(model,
                                        tokenizer,
                                        pixel_values,
                                        num_patches_list=num_patches_list,
                                        questions=questions,
                                        generation_config=generation_config)
    return response, scores

def qwen2_inference(model, processor, image_paths):
    questions = ['What is the text in the image? Answer should be a single word.'] * len(image_paths)
    
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
                                   max_new_tokens=32,
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
    questions = ['What is the text in the image? Answer should be a single word.'] * len(image_paths)
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

    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.pcolormesh(PHI, RHO, probabilities, cmap='plasma', vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(json_file.replace(".json", ".pdf"))
    plt.close()

    dr = rho[1] - rho[0]
    dphi = phi[1] - phi[0]
    sac = 0.0
    for r, p, prob in zip(RHO.flatten(), PHI.flatten(), probabilities.flatten()):
        sac += r * dr * dphi * prob
    
    with open(json_file.replace(".json", ".txt"), "w") as f:
        f.write(f"Sensitivity Area of Color (SAC): {sac}")
    print(f"Sensitivity Area of Color (SAC): {sac}")

def size_json2csv(json_file):
    csv_file = json_file.replace(".json", ".csv")

    with open(json_file, "r") as f:
        json_objects = json.load(f)
    
    font_sizes = np.linspace(1, 100, 1000)
    probability = [object['probability'] for object in json_objects]
    assert len(probability) == len(font_sizes)
    
    sum_prob = sum(probability)
    avg_prob = sum_prob / len(probability)
    print(f"Average Acc: {avg_prob}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="OpenGVLab/InternVL2_5-8B")
    parser.add_argument("--save_file", type=str, default="outputs/sensitivity/InternVL2_5-8B_eccentricity.json")
    parser.add_argument("--image_dir", type=str, default="dataset/sensitivity/eccentricity/shape-eccentricity")
    parser.add_argument("--bs", type=int, default=10)
    args = parser.parse_args()
    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-4B --save_file outputs/sensitivity/ocr/InternVL2_5-4B_ocr_red.json --image_dir dataset/sensitivity/ocr-red/ --bs 10
    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-4B --save_file outputs/sensitivity/ocr/InternVL2_5-4B_ocr_blue.json --image_dir dataset/sensitivity/ocr-blue/ --bs 10
    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-4B --save_file outputs/sensitivity/ocr/InternVL2_5-4B_ocr_green.json --image_dir dataset/sensitivity/ocr-green/ --bs 10

    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-8B --save_file outputs/sensitivity/ocr/InternVL2_5-8B_ocr_red.json --image_dir dataset/sensitivity/ocr-red/ --bs 10
    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-8B --save_file outputs/sensitivity/ocr/InternVL2_5-8B_ocr_blue.json --image_dir dataset/sensitivity/ocr-blue/ --bs 10
    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-8B --save_file outputs/sensitivity/ocr/InternVL2_5-8B_ocr_green.json --image_dir dataset/sensitivity/ocr-green/ --bs 10

    # python -m sensitivity.ocr_main --model_id Qwen/Qwen2-VL-2B-Instruct --save_file outputs/sensitivity/ocr/Qwen2-VL-2B-Instruct_ocr_red.json --image_dir dataset/sensitivity/ocr-red/ --bs 10
    # python -m sensitivity.ocr_main --model_id Qwen/Qwen2-VL-2B-Instruct --save_file outputs/sensitivity/ocr/Qwen2-VL-2B-Instruct_ocr_blue.json --image_dir dataset/sensitivity/ocr-blue/ --bs 10
    # python -m sensitivity.ocr_main --model_id Qwen/Qwen2-VL-2B-Instruct --save_file outputs/sensitivity/ocr/Qwen2-VL-2B-Instruct_ocr_green.json --image_dir dataset/sensitivity/ocr-green/ --bs 10

    # python -m sensitivity.ocr_main --model_id Qwen/Qwen2-VL-7B-Instruct --save_file outputs/sensitivity/ocr/Qwen2-VL-7B-Instruct_ocr_red.json --image_dir dataset/sensitivity/ocr-red/ --bs 10
    # python -m sensitivity.ocr_main --model_id Qwen/Qwen2-VL-7B-Instruct --save_file outputs/sensitivity/ocr/Qwen2-VL-7B-Instruct_ocr_blue.json --image_dir dataset/sensitivity/ocr-blue/ --bs 10
    # python -m sensitivity.ocr_main --model_id Qwen/Qwen2-VL-7B-Instruct --save_file outputs/sensitivity/ocr/Qwen2-VL-7B-Instruct_ocr_green.json --image_dir dataset/sensitivity/ocr-green/ --bs 10

    # python -m sensitivity.ocr_main --model_id HuggingFaceTB/SmolVLM-Instruct --save_file outputs/sensitivity/ocr/SmolVLM-Instruct_ocr_red.json --image_dir dataset/sensitivity/ocr-red/ --bs 10
    # python -m sensitivity.ocr_main --model_id HuggingFaceTB/SmolVLM-Instruct --save_file outputs/sensitivity/ocr/SmolVLM-Instruct_ocr_blue.json --image_dir dataset/sensitivity/ocr-blue/ --bs 10
    # python -m sensitivity.ocr_main --model_id HuggingFaceTB/SmolVLM-Instruct --save_file outputs/sensitivity/ocr/SmolVLM-Instruct_ocr_green.json --image_dir dataset/sensitivity/ocr-green/ --bs 10

    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-4B --save_file outputs/sensitivity/ocr-text/InternVL2_5-4B_ocr_red.json --image_dir dataset/sensitivity/ocr-text-red/ --bs 10
    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-4B --save_file outputs/sensitivity/ocr-text/InternVL2_5-4B_ocr_blue.json --image_dir dataset/sensitivity/ocr-text-blue/ --bs 10
    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-4B --save_file outputs/sensitivity/ocr-text/InternVL2_5-4B_ocr_green.json --image_dir dataset/sensitivity/ocr-text-green/ --bs 10

    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-8B --save_file outputs/sensitivity/ocr-text/InternVL2_5-8B_ocr_red.json --image_dir dataset/sensitivity/ocr-text-red/ --bs 10
    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-8B --save_file outputs/sensitivity/ocr-text/InternVL2_5-8B_ocr_blue.json --image_dir dataset/sensitivity/ocr-text-blue/ --bs 10
    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-8B --save_file outputs/sensitivity/ocr-text/InternVL2_5-8B_ocr_green.json --image_dir dataset/sensitivity/ocr-text-green/ --bs 10

    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-2B --save_file outputs/sensitivity/ocr-size/InternVL2_5-2B_ocr_size.json --image_dir dataset/sensitivity/ocr-size/ --bs 10
    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-4B --save_file outputs/sensitivity/ocr-size/InternVL2_5-4B_ocr_size.json --image_dir dataset/sensitivity/ocr-size/ --bs 10
    # python -m sensitivity.ocr_main --model_id OpenGVLab/InternVL2_5-8B --save_file outputs/sensitivity/ocr-size/InternVL2_5-8B_ocr_size.json --image_dir dataset/sensitivity/ocr-size/ --bs 10
    # python -m sensitivity.ocr_main --model_id HuggingFaceTB/SmolVLM-Instruct --save_file outputs/sensitivity/ocr-size/SmolVLM-Instruct_ocr_size.json --image_dir dataset/sensitivity/ocr-size/ --bs 10
    # python -m sensitivity.ocr_main --model_id HuggingFaceTB/SmolVLM-Base --save_file outputs/sensitivity/ocr-size/SmolVLM-Base_ocr_size.json --image_dir dataset/sensitivity/ocr-size/ --bs 10
    # python -m sensitivity.ocr_main --model_id Qwen/Qwen2-VL-2B-Instruct --save_file outputs/sensitivity/ocr-size/Qwen2-VL-2B_ocr_size.json --image_dir dataset/sensitivity/ocr-size/ --bs 10
    # python -m sensitivity.ocr_main --model_id Qwen/Qwen2-VL-7B-Instruct --save_file outputs/sensitivity/ocr-size/Qwen2-VL-7B_ocr_size.json --image_dir dataset/sensitivity/ocr-size/ --bs 10

    # model, tokenizer = load_vlm(args.model_id)
    # image_list = get_image_list(args.image_dir)
    
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
    #         score = 1.0 if 'hello' in responses[j] else 0.0
    #         json_object = {}
    #         json_object["image_path"] = image_path
    #         json_object["probability"] = score
    #         json_objects.append(json_object)

    # with open(args.save_file, "w") as f:
    #     json.dump(json_objects, f)
    
    if "size" in args.image_dir:
        size_json2csv(args.save_file)
    else:
        color_json2plot(args.save_file)