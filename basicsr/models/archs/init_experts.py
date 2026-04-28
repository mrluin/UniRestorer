from basicsr.models.archs.restormer_arch import Restormer, TransformerBlock
from basicsr.models.archs.retinexformer_arch import RetinexFormer
from basicsr.models.archs.nafnet_arch import NAFNetLocal
from basicsr.models.archs.dehazeformer_arch import dehazeformer_l
from basicsr.models.archs.restormer_lora_arch import LoRARestormer


def init_experts_three_layers(l0_experts, l1_experts, l2_experts):
    l0_experts.append(Restormer(inp_channels=3,
                                 out_channels=3,
                                 dim=48,
                                 num_blocks=[4, 6, 6, 8],
                                 num_refinement_blocks=4,
                                 heads=[1, 2, 4, 8],
                                 ffn_expansion_factor=2.66,
                                 bias=False,
                                 LayerNorm_type="WithBias",
                                 dual_pixel_task=False))
    for i in range(4):
        l1_experts.append(Restormer(inp_channels=3,
                                     out_channels=3,
                                     dim=48,
                                     num_blocks=[4, 6, 6, 8],
                                     num_refinement_blocks=4,
                                     heads=[1, 2, 4, 8],
                                     ffn_expansion_factor=2.66,
                                     bias=False,
                                     LayerNorm_type="BiasFree",
                                     dual_pixel_task=False))

    l1_experts.append(dehazeformer_l())
    l1_experts.append(RetinexFormer(in_channels=3,
                                    out_channels=3,
                                    n_feat=40,
                                    stage=1,
                                    num_blocks=[1,2,2]))
    l1_experts.append(NAFNetLocal(width=64,
                                  enc_blk_nums=[1, 1, 1, 28],
                                  middle_blk_num=1,
                                  dec_blk_nums=[1, 1, 1, 1]))
    for i in range(3):
        l2_experts.append(Restormer(inp_channels=3,
                                    out_channels=3,
                                    dim=48,
                                    num_blocks=[4, 6, 6, 8],
                                    num_refinement_blocks=4,
                                    heads=[1, 2, 4, 8],
                                    ffn_expansion_factor=2.66,
                                    bias=False,
                                    LayerNorm_type="BiasFree",
                                    dual_pixel_task=False))
    for i in range(4):
        l2_experts.append(LoRARestormer(inp_channels=3,
                                        out_channels=3,
                                        dim=48,
                                        num_blocks=[ 4,6,6,8 ],
                                        num_refinement_blocks=4,
                                        heads=[ 1,2,4,8 ],
                                        ffn_expansion_factor=2.66,
                                        bias=False,
                                        LayerNorm_type="WithBias",
                                        dual_pixel_task=False,
                                        r=[8,8,8,8],
                                        lora_alpha=[16,16,16,16]))

    for i in range(10):
        l2_experts.append(Restormer(inp_channels=3,
                                    out_channels=3,
                                    dim=48,
                                    num_blocks=[4, 6, 6, 8],
                                    num_refinement_blocks=4,
                                    heads=[1, 2, 4, 8],
                                    ffn_expansion_factor=2.66,
                                    bias=False,
                                    LayerNorm_type="WithBias",
                                    dual_pixel_task=False))
    for i in range(2):
        l2_experts.append(dehazeformer_l())

    for i in range(2):
        l2_experts.append(RetinexFormer(in_channels=3,
                                        out_channels=3,
                                        n_feat=40,
                                        stage=1,
                                        num_blocks=[1, 2, 2]))

    for i in range(2):
        l2_experts.append(NAFNetLocal(width=64,
                                      enc_blk_nums=[1, 1, 1, 28],
                                      middle_blk_num=1,
                                      dec_blk_nums=[1, 1, 1, 1]))

    return l0_experts, l1_experts, l2_experts





