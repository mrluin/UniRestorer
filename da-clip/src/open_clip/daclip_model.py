from typing import Optional

import logging
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy

from transformers import CLIPVisionModel

from .transformer import (
    ControlTransformer
)
from .model import CLIP, CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


class DaCLIP(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.clip = clip_model
        self.visual = clip_model.visual

        self.visual_control = copy.deepcopy(clip_model.visual)
        self.visual_control.transformer = ControlTransformer(self.visual_control.transformer)
        self.logit_scale = copy.deepcopy(clip_model.logit_scale)

        # self.visual_control.conv1 = nn.Conv2d(in_channels=6, out_channels=1024,
        #                                       kernel_size=14, stride=14, bias=False)

        # self.image_content_mapping = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        # )
        #
        # self.image_degra_mapping = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        #     nn.Linear(768, 768),
        #     nn.GELU(),
        # )

    def initial_controller(self):
        for (kv, param_v), (kc, param_c) in zip(self.clip.visual.named_parameters(), self.visual_control.named_parameters()):

            if 'transformer' not in kv:
                param_c.data.copy_(param_v.data)

            # if 'transformer' not in kv and 'conv1' not in kv:
            #     print(kv)
            #     param_c.data.copy_(param_v.data)

        for param_v, param_c in zip(self.clip.visual.transformer.parameters(), self.visual_control.transformer.parameters()):
            param_c.data.copy_(param_v.data)

        self.logit_scale.data.copy_(self.clip.logit_scale.data)
        
    def lock_clip(self):
        for param in self.clip.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.clip.visual.set_grad_checkpointing(enable)
        self.clip.transformer.grad_checkpointing = enable
        self.visual_control.set_grad_checkpointing(enable)

    # def encode_image(self, images, gt_images, control=False, normalize: bool = False):
    def encode_image(self, images, control=False, normalize: bool = False):
        if control:
            # concate_image = torch.cat([images, gt_images], dim=1)
            degra_features = self.visual_control(images, output_hiddens=False)

            # image_features = self.clip.visual(image, control=hiddens, output_hiddens=False)

            # print(image_features.shape)

            degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
            # image_features = F.normalize(image_features, dim=-1) if normalize else image_features

            # return image_features, degra_features
            return degra_features
        else:
            return self.clip.encode_image(image, normalize)

    def encode_text(self, text, normalize: bool = False):
        return self.clip.encode_text(text, normalize)

    def forward(
            self,
            images: Optional[torch.Tensor] = None,
            # gt_images: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        # (caption, degradation) = text.chunk(2, dim=-1) if text is not None else (None, None)

        degradation = text if text is not None else None

        # image_features, image_degra_features = self.encode_image(image, control=True, normalize=True) if image is not None else None
        # image_features_global, image_degra_features_global = self.encode_image(image[0], control=True, normalize=True) if image is not None else None
        #
        # x_query_lq = images[:, 0, ...]  # B C H W

        # image_degra_features0 = self.encode_image(images, gt_images, control=True, normalize=True) #if image is not None else None

        image_degra_features0 = self.encode_image(images, control=True, normalize=True) #if image is not None else None

        # image_features1, image_degra_features1 = self.encode_image(image[1], control=True, normalize=True) if image is not None else None
        # image_features2, image_degra_features2 = self.encode_image(image[2], control=True, normalize=True) if image is not None else None
        # image_features3, image_degra_features3 = self.encode_image(image[3], control=True, normalize=True) if image is not None else None
        # image_features4, image_degra_features4 = self.encode_image(image[4], control=True, normalize=True) if image is not None else None

        # image_features5, image_degra_features5 = self.encode_image(image[5], control=True,
        #                                                            normalize=True) if image is not None else None
        # image_features6, image_degra_features6 = self.encode_image(image[6], control=True,
        #                                                            normalize=True) if image is not None else None
        # image_features7, image_degra_features7 = self.encode_image(image[7], control=True,
        #                                                            normalize=True) if image is not None else None

        # image_features = torch.stack([image_features0, image_features1, image_features2, image_features3, image_features4], dim=1)  # B 5, 512
        # image_degra_features = torch.stack([image_degra_features0, image_degra_features1, image_degra_features2, image_degra_features3, image_degra_features4],  dim=1) # B, 5, 512

        # image_features = torch.mean(image_features, dim=1)
        # image_degra_features = torch.mean(image_degra_features, dim=1)

        # image_features = torch.cat([image_features_global, image_features], dim=1)
        # image_degra_features = torch.cat([image_degra_features_global, image_degra_features], dim=1)

        # image_features = self.image_content_mapping(image_features)
        # image_degra_features = self.image_degra_mapping(image_degra_features)

        # print("after", image_features.dtype)   # float16

        # text_features = self.encode_text(caption, normalize=True) if text is not None else None
        text_degra_features = self.encode_text(degradation, normalize=True) if degradation is not None else None

        return {
            # "image_features": image_features,
            # "text_features": text_features,
            "image_degra_features": image_degra_features0,
            "text_degra_features": text_degra_features,
            "logit_scale": self.logit_scale.exp()
        }

