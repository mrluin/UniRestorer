import glob
import os
import torch
import numpy as np
import cv2

from basicsr.utils.img_util import imfrombytes, img2tensor, tensor2img, imwrite

from basicsr.models.archs.unirestorer_arch import UniRestorer
from PIL import Image

import argparse
import open_clip


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--lq_path", type=str, required=True)
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()

    daclip_checkpoint = './checkpoints/daclip.pt'
    daclip_model, process = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=daclip_checkpoint)

    model = UniRestorer(
        daclip_model, embed_dim=512, num_feat=48, topk=1, num_experts=[1, 7, 22]
    )
    model.load_state_dict(torch.load("./checkpoints/unirestorer_all_in_one.pth", weights_only=True)["params"])
    model = model.to('cuda')
    model.eval()

    # inference
    img_name = args.lq_path.split('/')[-1]

    f = open(args.lq_path, "rb")
    img_bytes = f.read()
    lq_img = imfrombytes(img_bytes, float32=True)
    daclip_lq = process(Image.open(args.lq_path)).unsqueeze(0).to('cuda')
    lq_img = img2tensor(lq_img).unsqueeze(0).to('cuda')

    with torch.no_grad():
        pred, _, _, _ = model(lq_img, daclip_lq)
        pred = tensor2img(pred)
        imwrite(pred, os.path.join(args.save_path, img_name))