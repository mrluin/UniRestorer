import glob
import os
import torch
import numpy as np
import cv2
from sympy.codegen.ast import float32

from basicsr.utils.img_util import imfrombytes, img2tensor, tensor2img, single2uint, uint2single, imread_uint, imwrite
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

from basicsr.models.archs.uncertainty_moe_033_offline_arch import MioRestorer_UncertaintyMoE_V1
from PIL import Image

import argparse
import open_clip

from tqdm import tqdm

task_dict = {
    '0': 'LightNoise',
    '1': 'MediumNoise',
    '2': 'HeavyNoise',
    '3': 'LightCompression',
    '4': 'MediumCompression',
    '5': 'HeavyCompression',
    '6': 'LightSnow',  #
    '7': 'MediumSnow',  #
    '8': 'HeavySnow',    #
    '9': 'DDNLightRain',  #
    '10': 'DDNLargeRain',  #
    '11': 'DIDLightRain',  #
    '12': 'DIDMediumRain',  #
    '13': 'DIDHeavyRain',  #
    '14': 'Rain200H',  #
    '15': 'Rain200L',   #
    '16': 'LightHaze',  #
    '17': 'HeavyHaze',  #
    '18': 'LightLowlight',  #
    '19': 'HeavyLowlight',  #
    '20': 'LightBlur',  #
    '21': 'HeavyBlur',  #
}

expert_dict = {
    '0': 'All-in-One',
    '1': 'Denoise',
    '2': 'CAR',
    '3': 'Desnow',
    '4': 'Drain',
    '5': 'Dehaze',
    '6': 'Lowlight',  #
    '7': 'Deblur',  #
    '8': 'Denoise-light',    #
    '9': 'Denoise-medium',  #
    '10': 'Denoise-heavy',  #
    '11': 'CAR-light',  #
    '12': 'CAR-medium',  #
    '13': 'CAR-heavy',  #
    '14': 'Snow-S',  #
    '15': 'Snow-M',   #
    '16': 'Snow-L',  #
    '17': 'DDNLightRain',  #
    '18': 'DDNLargeRain',  #
    '19': 'DIDLightRain',  #
    '20': 'DIDMediumRain',  #
    '21': 'DIDHeavytRain',  #
    '22': 'Rain200H',  #
    '23': 'Rain200L',  #
    '24': 'LightHaze',  #
    '25': 'HeavyHaze',  #
    '26': 'LightDark',  #
    '27': 'HeavyDark',  #
    '28': 'LightBlur',  #
    '29': 'HeavyBlur',  #
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--lq_folder", type=str)
    parser.add_argument('--gt_folder', type=str)
    parser.add_argument("--save_images", action="store_true")

    parser.add_argument("--degree", type=float)

    args = parser.parse_args()

    # task_list = ["dejpeg"]
    # task_list = ["derain", "dehaze", "denoise", "deblur", "low-light", "desnow", "dejpeg"]
    # task_list = ["denoise", "deblur", "low-light", "desnow", "dejpeg"]
    # save_root = "/hdd1/linjingbo/code/Restormer-main/results/Restormer_all-in-one"
    # checkpoint_path = "/hdd1/linjingbo/code/Restormer-main/experiments/001_Restormer_all-in-one_train_from_scratch/models/net_g_300000.pth"

    daclip_checkpoint = '/home/linjingbo/codes/daclip-uir-main/da-clip/src/logs/daclip_ViT-B-32_b784x1_lr2e-5_e100_joint_animal/checkpoints/epoch_100.pt'
    daclip_model, process = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=daclip_checkpoint)

    model = MioRestorer_UncertaintyMoE_V1(
        daclip_model, embed_dim=512, num_feat=48, topk=1, num_experts=[1, 7, 22]
    )
    model.load_state_dict(torch.load("./experiments/008_Ours_ver3/models/net_g_10000.pth", weights_only=True)["params"])
    # model.load_state_dict(torch.load("./experiments/061_Ours_ver3_train_sigma/models/net_g_10000.pth", weights_only=True)["params"])
    model = model.to('cuda')
    model.eval()

    # Derain Rain100L
    # lq_root_derain = "/media/linjingbo/HDD/linjingbo/data/Rain100L/test/input"
    # gt_root_derain = "/media/linjingbo/HDD/linjingbo/data/Rain200H/test/target"
    #
    # # Dehaze sots-outdoor
    # lq_root_dehaze = "/home/linjingbo/data/SOTS-outdoor/hazy"
    # gt_root_dehaze = "/home/linjingbo/data/SOTS-outdoor/gt"
    #
    # # gaussian denoise 15 25 50, average
    # # gt_root_denoise_urban100 = "/hdd2/linjingbo/urban100"
    # gt_root_denoise = "/home/linjingbo/data/CBSD68"
    #
    # # motion blur
    # lq_root_deblur = "/home/linjingbo/data/GoPro/test/input"
    # gt_root_deblur = "/home/linjingbo/data/GoPro/test/target"
    #
    # # low-light enh
    # lq_root_dark = "/home/linjingbo/data/LOLv1/Test/input"
    # gt_root_dark = "/home/linjingbo/data/LOLv1/Test/target"
    #
    # # desnow Snow100K-L
    # lq_root_desnow = "/home/linjingbo/data/Snow100K-L/synthetic"
    # gt_root_desnow = "/home/linjingbo/data/Snow100K-L/gt"
    #
    # # compression artifacts removal, p=10, p=20, p=30, p=40, average
    # gt_root_jpeg = "/home/linjingbo/data/LIVE1"

    if args.task_name in ['Rain200H', 'Rain200L', 'DDN', 'DID',
                          'GoPro', 'LOLv1', 'Snow100K_L', 'Snow100K_S', 'Snow100K_M']:
        lq_paths = sorted(glob.glob(os.path.join(args.lq_folder, "*")))
        gt_paths = sorted(glob.glob(os.path.join(args.gt_folder, "*")))
    elif args.task_name == "Rain100L":
        lq_paths = sorted(glob.glob(os.path.join(args.lq_folder, "*")))
        gt_paths = [os.path.join(args.gt_folder, f"no{path_.split('/')[-1]}") for path_ in lq_paths]
    elif args.task_name == "SOTS_outdoor":
        lq_paths = sorted(glob.glob(os.path.join(args.lq_folder, '*')))
        gt_paths = [os.path.join(args.gt_folder, f"{lq_path_.split('/')[-1].split('_')[0]}.png") for lq_path_ in lq_paths]
    elif args.task_name in ['Denoise', 'CAR']:
        lq_paths = None
        gt_paths = sorted(glob.glob(os.path.join(args.gt_folder, '*')))
        lq_paths = gt_paths
    else:
        raise NotImplementedError

    psnr_list = []
    ssim_list = []


    cnt = 0

    for lq_path, gt_path in tqdm(zip(lq_paths, gt_paths)):

        if args.task_name == 'Denoise':

            img_name = gt_path.split('/')[-1].split('.')[0]

            noise_level = torch.FloatTensor([args.degree]) / 255.0

            f = open(gt_path, "rb")
            img_bytes = f.read()
            gt_img = imfrombytes(img_bytes, float32=True)

            lq_img = gt_img.copy()
            lq_img = img2tensor(lq_img).unsqueeze(0).to("cuda")
            gt_img = img2tensor(gt_img)

            noise = torch.randn(lq_img.size()).mul_(noise_level).float().to('cuda')
            lq_img.add_(noise)

            daclip_lq = Image.fromarray(tensor2img(lq_img))
            daclip_lq = process(daclip_lq).unsqueeze(0).to('cuda')

            img_name = f"{img_name}_{args.degree}"


        elif args.task_name == "CAR":

            p = int(args.degree)
            img_name = gt_path.split('/')[-1].split('.')[0]
            lq_img = imread_uint(gt_path, 3)  # RGB unit

            gt_img = lq_img.copy()  # RGB  uint
            lq_img = cv2.cvtColor(lq_img, cv2.COLOR_RGB2BGR) # BGR
            result, encimg = cv2.imencode('.jpg', lq_img, [int(cv2.IMWRITE_JPEG_QUALITY), p])
            lq_img = cv2.imdecode(encimg, 1)  # BGR uint
            lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)  # RGB uint

            daclip_lq = process(Image.fromarray(lq_img)).unsqueeze(0).to("cuda")

            lq_img = img2tensor(lq_img, bgr2rgb=False).unsqueeze(0).to("cuda")  # RGB float
            gt_img = img2tensor(gt_img, bgr2rgb=False)  # RGB float

            # lq_img = uint2single(lq_img)
            # lq_img = imread_uint(gt_path, 3)
            # gt_img = lq_img.copy()
            # lq_img = cv2.cvtColor(lq_img, cv2.COLOR_RGB2BGR)
            # result, encimg = cv2.imencode('.jpg', lq_img, [int(cv2.IMWRITE_JPEG_QUALITY), p])
            # lq_img = cv2.imdecode(encimg, 1)
            # lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)

            img_name = f"{img_name}_{args.degree}"

        else:

            img_name = lq_path.split('/')[-1]

            f = open(lq_path, "rb")
            img_bytes = f.read()
            lq_img = imfrombytes(img_bytes, float32=True)

            f= open(gt_path, "rb")
            img_bytes = f.read()
            gt_img = imfrombytes(img_bytes, float32=True)

            daclip_lq = process(Image.open(lq_path)).unsqueeze(0).to('cuda')

            lq_img = img2tensor(lq_img).unsqueeze(0).to('cuda')
            gt_img = img2tensor(gt_img)

        # lq_img = tensor2img(lq_img)
        # imwrite(lq_img, os.path.join("./results/Denoise", img_name))

        with torch.no_grad():
            preds, mu, logvar, lf_logits, layer_logits, lf_clean_logits, layer_clean_logits, gate_output, expert_indices, \
                lf_probs, layer_probs, layer_topk_indices, \
                aux_loss = model(lq_img, daclip_lq)

        class_index = torch.argmax(lf_probs).cpu().numpy()
        expert_indices = expert_indices.cpu().numpy()

        preds = tensor2img(preds)
        gt_img = tensor2img(gt_img)

        if args.save_images:

            if args.task_name == "Rain100L":
                save_task_name = "Derain"
            elif args.task_name == "GoPro":
                save_task_name = "Deblurring"
            elif args.task_name == "SOTS_outdoor":
                save_task_name = "Dehaze"
            elif args.task_name == "CAR":
                save_task_name = "Dejpeg"
            elif args.task_name == "Denoise":
                save_task_name = "Denoise"
            elif args.task_name == "LOLv1":
                save_task_name = "Lowlight"
            elif args.task_name == "Snow100K_L":
                save_task_name = "Desnow"
                img_name = f"{cnt:06d}"

            save_root = f"/media/linjingbo/HDD/linjingbo/suppl_visual_comparisons/All_in_One/{save_task_name}/Ours"
            save_path = os.path.join(save_root, f"{img_name}.png")
            imwrite(preds, save_path)

            cnt += 1

        psnr_ = calculate_psnr(preds, gt_img, crop_border=0, test_y_channel=True if 'Rain' in args.task_name else False)
        # ssim_ = calculate_ssim(preds, gt_img, crop_border=0, test_y_channel=True if 'Rain' in args.task_name else False)

        psnr_list.append(psnr_)
        # ssim_list.append(ssim_)

        # print(f"# LQ_path {lq_path}: Task {args.task_name} Pred {task_dict[f'{str(class_index)}']} Expert {expert_dict[f'{str(expert_indices[0])}']} PSNR {psnr_} SSIM {ssim_}")

    # print(f"# Task {args.task_name}, Average {np.asarray(psnr_list).mean()} {np.asarray(ssim_list).mean()}")
    print(f"# Task {args.task_name}, Average {np.asarray(psnr_list).mean()}")

# Rain100L  43.033 0.9911
# Rain200L  41.522 0.9886
# Rain200H  31.822 0.9305
# DID  33.689 0.9575
# DDN  31.603 0.9517

# urban100
# Noise 15 33.891 0.9589
# Noise 25 32.199 0.9476
# Noise 50 29.754 0.9304

# cbsd68
# Noise 15  34.037 0.9645
# Noise 25  31.706 0.9456
# Noise 50  28.528 0.8999

# LIVE
# JPEG 10  27.890 0.8788
# JPEG 20  30.314 0.9270
# JPEG 30  31.653 0.9448
# JPEG 40  32.582 0.9545

# GOPRO

# LOLv1   26.669 0.8629

# Snow100k-L
# Snow100k-m
# Snow100k-s

# haze
