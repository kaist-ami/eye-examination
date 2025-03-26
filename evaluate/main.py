import os
import json
import argparse
import torch
from utils import load_image, load_vlm
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

import pdb

def internvl25_inference(model, tokenizer, image_path):
    if 'color' in image_path and 'yes_or_no' in image_path:
        question = '<image>\n Do the two samples have the same color? Answer with yes or no.'
    elif 'color' in image_path and '1_or_2' in image_path:
        question = '<image>\n Each sample has two colored boxes, which sample has the same colored boxes? Answer with sample 1, sample 2, or No answer.'
    elif 'shape' in image_path and 'yes_or_no' in image_path:
        question = '<image>\n Are the two samples identical in shape? Answer with yes or no.'
    elif 'shape' in image_path and '1_or_2' in image_path:
        question = '<image>\n  Each sample has two shapes, offer an answer which sample has the sameshape. Answer with sample 1, sample 2, or No answer.'

    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=128, do_sample=False, return_dict_in_generate=True, output_scores=True, pad_token_id=tokenizer.pad_token_id)
    response, scores = model.chat(model, tokenizer, pixel_values, question, generation_config)
    return response, scores

def smolvlm_inference(model, processor, image_path):
    image = load_image(image_path, model_id="SmolVLM")
    if 'color' in image_path and 'yes_or_no' in image_path:
        question = 'Do the two samples have the same color? Answer with yes or no.'
    elif 'color' in image_path and '1_or_2' in image_path:
        question = 'Each sample has two colored boxes, which sample has the same colored boxes? Answer with sample 1, sample 2, or No answer.'
    elif 'shape' in image_path and 'yes_or_no' in image_path:
        question = 'Are the two samples identical in shape? Answer with yes or no.'
    elif 'shape' in image_path and '1_or_2' in image_path:
        question = 'Each sample has two shapes, offer an answer which sample has the sameshape. Answer with sample 1, sample 2, or No answer.'
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda')

    generated_ids = model.generate(**inputs, 
                                   max_new_tokens=128,
                                   do_sample=False,
                                   return_dict_in_generate=True,
                                   output_scores=True)
    sequences = generated_ids.sequences
    response = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    response = response[0].split('Assistant: ')[-1]
    scores = generated_ids.scores[0]
    return response, scores

def qwen2_inference(model, processor, image_path):
    if 'color' in image_path and 'yes_or_no' in image_path:
        question = 'Do the two samples have the same color? Answer with yes or no.'
    elif 'color' in image_path and '1_or_2' in image_path:
        question = 'Each sample has two colored boxes, which sample has the same colored boxes? Answer with sample 1, sample 2, or No answer.'
    elif 'shape' in image_path and 'yes_or_no' in image_path:
        question = 'Are the two samples identical in shape? Answer with yes or no.'
    elif 'shape' in image_path and '1_or_2' in image_path:
        question = 'Each sample has two shapes, offer an answer which sample has the sameshape. Answer with sample 1, sample 2, or No answer.'
    
    messages = [
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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, 
                                   max_new_tokens=128,
                                   do_sample=False,
                                   return_dict_in_generate=True,
                                   output_scores=True)
    sequences = generated_ids.sequences
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, sequences)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    scores = generated_ids.scores[0]
    return response, scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="OpenGVLab/InternVL2_5-8B")
    parser.add_argument("--image_dir", type=str, default="dataset/colors/yes_or_no/")
    parser.add_argument("--image_json", type=str, default="test.json")
    args = parser.parse_args()

    model, tokenizer = load_vlm(args.model_id)

    json_file = os.path.join(args.image_dir, args.image_json)
    with open(json_file, "r") as f:
        data = json.load(f)
    
    correct = 0.0
    for i in tqdm(range(len(data)), total=len(data)):
        image_path = data[i]['image']
        image_path = os.path.join(args.image_dir, image_path)
        answer = data[i]['conversations'][-1]['value'].lower()

        if 'InternVL2_5' in args.model_id:
            response, scores = internvl25_inference(model, tokenizer, image_path)
        elif 'SmolVLM' in args.model_id:
            response, scores = smolvlm_inference(model, tokenizer, image_path)
        elif 'Qwen2-VL' in args.model_id:
            response, scores = qwen2_inference(model, tokenizer, image_path)
        else:
            raise NotImplementedError
        
        response = response.lower()
        if answer in response:
            correct += 1
    
    accuracy = correct / len(data)
    print(f"Accuracy: {accuracy}")
    
        