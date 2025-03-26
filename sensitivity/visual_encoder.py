
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

@torch.no_grad()
def internvl25_inference(model, tokenizer, ref_color, target_colors):
    ref_pixel_value = load_image(ref_color, max_num=12).to(torch.bfloat16).cuda()
    target_pixel_values = [load_image(target_color, max_num=12).to(torch.bfloat16).cuda() for target_color in target_colors]
    target_pixel_values = torch.cat(target_pixel_values)

    ref_vit_embeds = model.vision_model(ref_pixel_value,
                                        output_hidden_states=False,
                                        return_dict=True).last_hidden_state
    ref_vit_embeds = ref_vit_embeds[:, 1:, :]
    h = w = int(ref_vit_embeds.shape[1] ** 0.5)
    ref_vit_embeds = ref_vit_embeds.reshape(ref_vit_embeds.shape[0], h, w, -1)
    ref_vit_embeds = model.pixel_shuffle(ref_vit_embeds, scale_factor=model.downsample_ratio)
    ref_vit_embeds = ref_vit_embeds.reshape(ref_vit_embeds.shape[0], -1, ref_vit_embeds.shape[-1])
    ref_vit_embeds = model.mlp1(ref_vit_embeds).mean(dim=1)
    
    # ref_vit_embeds = torch.nn.functional.normalize(ref_vit_embeds[:, 1:, :], p=2, dim=-1).mean(dim=1)
    target_vit_embeds = model.vision_model(target_pixel_values,
                                           output_hidden_states=False,
                                           return_dict=True).last_hidden_state
    target_vit_embeds = target_vit_embeds[:, 1:, :]
    h = w = int(target_vit_embeds.shape[1] ** 0.5)
    target_vit_embeds = target_vit_embeds.reshape(target_vit_embeds.shape[0], h, w, -1)
    target_vit_embeds = model.pixel_shuffle(target_vit_embeds, scale_factor=model.downsample_ratio)
    target_vit_embeds = target_vit_embeds.reshape(target_vit_embeds.shape[0], -1, target_vit_embeds.shape[-1])
    target_vit_embeds = model.mlp1(target_vit_embeds).mean(dim=1)
    # target_vit_embeds = torch.nn.functional.normalize(target_vit_embeds[:, 1:, :], p=2, dim=-1).mean(dim=1)
    scores = torch.nn.functional.cosine_similarity(ref_vit_embeds, target_vit_embeds, dim=1)
    return scores

@torch.no_grad()
def internvit_inference(model, tokenizer, ref_color, target_colors):
    ref_pixel_value = load_image(ref_color, model_id="InternViT")
    target_pixel_values = [load_image(target_color, model_id="InternViT") for target_color in target_colors]

    ref_pixel_value = tokenizer(images=ref_pixel_value, return_tensors="pt").pixel_values
    ref_pixel_value = ref_pixel_value.to(torch.bfloat16).cuda()
    target_pixel_values = tokenizer(images=target_pixel_values, return_tensors="pt").pixel_values
    target_pixel_values = target_pixel_values.to(torch.bfloat16).cuda()

    ref_feature = model(ref_pixel_value).pooler_output
    target_features = model(target_pixel_values).pooler_output

    # cosine similarity between ref_feature and target_features
    scores = torch.nn.functional.cosine_similarity(ref_feature, target_features, dim=1)
    return scores
    

@torch.no_grad()
def clip_inference(model, tokenizer, ref_color, target_colors):
    # ref_pixel_value = load_image(ref_color, model_id="clip")
    ref_text = f"a photo of {ref_color}"
    target_pixel_values = [load_image(target_color, model_id="clip") for target_color in target_colors]

    # ref_pixel_value = tokenizer(images=ref_pixel_value, return_tensors="pt")
    ref_text_value = tokenizer(ref_text, padding=True, return_tensors="pt")
    target_pixel_values = tokenizer(images=target_pixel_values, return_tensors="pt")

    # ref_pixel_value = {k: v.cuda() for k, v in ref_pixel_value.items()}
    ref_text_value = {k: v.cuda() for k, v in ref_text_value.items()}
    target_pixel_values = {k: v.cuda() for k, v in target_pixel_values.items()}

    # ref_feature = model.get_image_features(**ref_pixel_value)
    ref_feature = model.get_text_features(**ref_text_value)
    target_features = model.get_image_features(**target_pixel_values)
    # cosine similarity between ref_feature and target_features
    scores = torch.nn.functional.cosine_similarity(ref_feature, target_features, dim=1)
    return scores

@torch.no_grad()
def siglip_inference(model, tokenizer, ref_color, target_colors):
    ref_text = f"a photo of {ref_color}"
    target_pixel_values = [load_image(target_color, model_id="clip") for target_color in target_colors]

    ref_text_value = tokenizer(ref_text, padding="max_length", return_tensors="pt")
    target_pixel_values = tokenizer(images=target_pixel_values, return_tensors="pt")

    ref_text_value = {k: v.cuda() for k, v in ref_text_value.items()}
    target_pixel_values = {k: v.cuda() for k, v in target_pixel_values.items()}

    ref_feature = model.get_text_features(**ref_text_value)
    target_features = model.get_image_features(**target_pixel_values)
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
    parser.add_argument("--model_id", type=str, default="openai/clip-vit-large-patch14-336")
    # openai/clip-vit-large-patch14-336
    # google/siglip-so400m-patch14-384
    parser.add_argument("--save_file", type=str, default="outputs/sensitivity/color_visual_encoder/clip-vit-large-patch14-336.json")
    parser.add_argument("--color", type=str, default="red")
    parser.add_argument("--bs", type=int, default=1000)
    args = parser.parse_args()

    # if args.color == "red":
    #     ref_color = (255, 0, 0)
    # elif args.color == "green":
    #     ref_color = (0, 255, 0)
    # elif args.color == "blue":
    #     ref_color = (0, 0, 255)
    

    # rho = np.linspace(0, 1, 100)
    # phi = np.linspace(0, math.pi * 2, 500)
    # RHO, PHI = np.meshgrid(rho, phi)pymnv
    # h = (PHI-PHI.min()) / (PHI.max()-PHI.min()) # use angle to determine hue, normalized from 0-1
    # h = np.flip(h)
    # s = RHO               # saturation is set as a function of radias
    # v = np.ones_like(RHO) # value is constant
    # h,s,v = h.flatten().tolist(), s.flatten().tolist(), v.flatten().tolist()
    # target_colors = [colorsys.hsv_to_rgb(*x) for x in zip(h,s,v)]
    # target_colors = np.array(target_colors)
    # target_colors = (target_colors * 255).astype(np.uint8)

    # model, tokenizer = load_vlm(args.model_id)
    # json_objects = []
    # count = 0
    # for i in tqdm(range(0, len(target_colors), args.bs), total=len(target_colors)//args.bs):
    #     # ref_color and target_color to image
    #     ref_color_img = np.ones((224, 224, 3), dtype=np.uint8) * ref_color
    #     bs = min(args.bs, len(target_colors[i:i+args.bs]))
    #     target_color_imgs = np.ones((bs, 224, 224, 3), dtype=np.uint8) * target_colors[i:i+args.bs].reshape(-1, 1, 1, 3)
        
    #     if "InternVL2_5" in args.model_id:
    #         scores = internvl25_inference(model, tokenizer, ref_color_img, target_color_imgs)
    #     elif "clip" in args.model_id:
    #         # scores = clip_inference(model, tokenizer, ref_color_img, target_color_imgs)
    #         scores = clip_inference(model, tokenizer, args.color, target_color_imgs)
    #     elif "InternViT" in args.model_id:
    #         scores = internvit_inference(model, tokenizer, ref_color_img, target_color_imgs)
    #     elif "siglip" in args.model_id:
    #         scores = siglip_inference(model, tokenizer, args.color, target_color_imgs)

    #     for j, target_color in enumerate(target_colors[i:i+args.bs]):
    #         json_object = {}
    #         json_object["probability"] = scores[j].item()
    #         json_objects.append(json_object)

    save_file = args.save_file.replace(".json", f"_{args.color}.json")
    # with open(save_file, "w") as f:
    #     json.dump(json_objects, f)
    
    color_json2plot(save_file)