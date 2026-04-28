import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.init_experts import init_experts_three_layers

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        inp_exp = inp[self._batch_index].squeeze(1)

        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, gates, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)
        stitched_gates = torch.cat(gates, 0)

        if multiply_by_gates:
            stitched = stitched * stitched_gates[:, :, None, None]
        else:
            stitched = stitched * (stitched_gates / stitched_gates.detach())[:, :, None, None]

        zeros = torch.zeros(self._gates.size(0),
                            expert_out[-1].size(1),
                            expert_out[-1].size(2),
                            expert_out[-1].size(3),
                            requires_grad=True,
                            device=stitched.device)  # B,

        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())

        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class UniRestorer(nn.Module):
    def __init__(self,
                 daclip_model,
                 embed_dim=512,
                 num_feat=48,
                 topk=1,
                 num_experts=[1, 7, 22],
                 ckpt_path_list=None,
                 noisy_topk_gating=True,
                 gating_activation=None,
                 cv_loss_weight=0.01,
                 switch_loss_weight=0.,
                 z_loss_weight=0.,
                 ):
        super(UniRestorer, self).__init__()

        # degra embedding extractor
        self.daclip_model = daclip_model

        # general options and settings
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.ckpt_path_list = ckpt_path_list
        self.topk = topk
        self.num_feat = num_feat

        self.noisy_topk_gating = noisy_topk_gating

        self.num_lf_nodes = self.num_experts[-1]
        self.num_layers = len(self.num_experts)
        self.is_scale_prob = True if self.topk != 1 else False

        self.cv_loss_weight = cv_loss_weight
        self.switch_loss_weight = switch_loss_weight
        self.z_loss_weight = z_loss_weight

        # expert networks, pretrained
        self.experts = []
        self.l0_experts = nn.ModuleList()
        self.l1_experts = nn.ModuleList()
        self.l2_experts = nn.ModuleList()

        self.experts = [self.l0_experts, self.l1_experts, self.l2_experts]

        #
        self.l0_experts, self.l1_experts, self.l2_experts = init_experts_three_layers(self.l0_experts, self.l1_experts, self.l2_experts)


        self.expert_pool = [self.l0_experts[0],
                            self.l1_experts[0], self.l1_experts[1], self.l1_experts[2], self.l1_experts[3], self.l1_experts[4], self.l1_experts[5], self.l1_experts[6],
                            self.l2_experts[0], self.l2_experts[1], self.l2_experts[2], self.l2_experts[3], self.l2_experts[4], self.l2_experts[5], self.l2_experts[6], self.l2_experts[7],
                            self.l2_experts[8], self.l2_experts[9], self.l2_experts[10], self.l2_experts[11], self.l2_experts[12], self.l2_experts[13], self.l2_experts[14], self.l2_experts[15],
                            self.l2_experts[16], self.l2_experts[17], self.l2_experts[18], self.l2_experts[19], self.l2_experts[20], self.l2_experts[21]
                            ]

        self.expert_indices_dict = torch.tensor([[0, 1, 8],  # 0
                                                 [0, 1, 9],   # 1
                                                 [0, 1, 10],  # 2
                                                 [0, 2, 11],  # 3
                                                 [0, 2, 12],  # 4
                                                 [0, 2, 13],  # 5
                                                 [0, 3, 14],  # 6
                                                 [0, 3, 15],  # 7
                                                 [0, 3, 16],  # 8
                                                 [0, 4, 17],  # 9
                                                 [0, 4, 18],  # 10
                                                 [0, 4, 19],  # 11
                                                 [0, 4, 20],  # 12
                                                 [0, 4, 21],  # 13
                                                 [0, 4, 22],  # 14
                                                 [0, 4, 23],  # 15
                                                 [0, 5, 24],  # 16
                                                 [0, 5, 25],  # 17
                                                 [0, 6, 26],  # 18
                                                 [0, 6, 27],  # 19
                                                 [0, 7, 28],  # 20
                                                 [0, 7, 29]   # 21
                                             ], dtype=torch.long).to("cuda")

        # leaf node router body
        self.mu = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512)

        )

        self.sigma = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512)
        )

        self.gamma = nn.Parameter(torch.ones(1) * 1e-4)
        self.beta = nn.Parameter(torch.ones(1) * (-7))

        self.lf_gate = nn.Sequential(
            nn.Linear(512,
                      2 * self.num_lf_nodes if self.noisy_topk_gating else self.num_lf_nodes,
                      bias=False)
        )
        self.layer_gate = nn.Sequential(
            nn.Linear(512,
                      2 * self.num_layers if self.noisy_topk_gating else self.num_layers,
                      bias=False)
        )

        nn.init.zeros_(self.layer_gate[-1].weight)
        nn.init.zeros_(self.lf_gate[-1].weight)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))


    def _init_route_layers_(self):

        nn.init.zeros_(self.layer_gate[-1].weight)

    def _init_experts_weight_(self):

        l0_ckpt_path_list = self.ckpt_path_list[:1]
        l1_ckpt_path_list = self.ckpt_path_list[1:8]
        l2_ckpt_path_list = self.ckpt_path_list[8:30]

        for expert, path in zip(self.l0_experts, l0_ckpt_path_list):
            expert.load_state_dict(torch.load(path, weights_only=True)["params"])

        for expert, path in zip(self.l1_experts, l1_ckpt_path_list):
            expert.load_state_dict(torch.load(path, weights_only=True)["params"])

        for expert, path in zip(self.l2_experts, l2_ckpt_path_list):
            expert.load_state_dict(torch.load(path, weights_only=True)["params"])

        print("Experts Loaded ...")

    def check_image_size(self, x):

        _, _, h, w = x.size()
        mod_pad_h = (self.scale_ratio - h % self.scale_ratio) % self.scale_ratio
        mod_pad_w = (self.scale_ratio - w % self.scale_ratio) % self.scale_ratio
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
           Useful as a loss to encourage a positive distribution to be more uniform.
           Epsilons added for numerical stability.
           Returns 0 for an empty Tensor.
           Args:
           x: a `Tensor`.
           Returns:
           a `Scalar`.
           """
        eps = 1e-10

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def compute_cvloss(self, probs, lf_indice=None):

        if lf_indice is not None:
            probs = torch.gather(probs, dim=1, index=lf_indice[:, :, None].expand(-1, -1, probs.shape[-1])).squeeze(1)

        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=-1))

    def compute_switchloss(self, probs, freqs, num_experts):

        loss = F.normalize(probs.sum(0), p=1, dim=-1) * F.normalize(freqs.float(), p=1, dim=-1)
        return loss.sum() * num_experts

    def compute_zloss(self, logits):

        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=-1)) ** 2)
        return zloss

    def get_sparse_logits(self, logits, topk_indices, topk_logits, eps=1e-10):

        zeros = torch.zeros_like(logits, requires_grad=True)
        lf_sparse_logits = zeros.scatter(-1, topk_indices, topk_logits)

        mask = lf_sparse_logits != 0

        return lf_sparse_logits, mask

    def topk_gating(self, mu, logvar,
                    layer_skip_mask=None, skip_mask=None,
                    sample_topk=0, noise_epsilon=1e-2):

        B, _ = mu.shape

        lf_clean_logits = self.lf_gate(mu)  # [B, num_leaf]
        layer_clean_logits = self.layer_gate(logvar)  # [B, num_layers]

        if self.noisy_topk_gating and self.training:
            lf_clean_logits, lf_raw_noise_stddev = lf_clean_logits.chunk(2, dim=-1)
            lf_noise_stddev = F.softplus(lf_raw_noise_stddev) + noise_epsilon
            lf_eps = torch.randn_like(lf_clean_logits)
            lf_noisy_logits = lf_clean_logits + lf_eps * lf_noise_stddev
            lf_logits = lf_noisy_logits
            layer_clean_logits, layer_raw_noise_stddev = layer_clean_logits.chunk(2, dim=-1)
            layer_noise_stddev = F.softplus(layer_raw_noise_stddev) + noise_epsilon
            layer_eps = torch.randn_like(layer_clean_logits)
            layer_noisy_logits = layer_clean_logits + layer_eps * layer_noise_stddev
            layer_logits = layer_noisy_logits
        elif self.noisy_topk_gating:
            lf_logits, _ = lf_clean_logits.chunk(2, dim=-1)
            layer_logits, _ = layer_clean_logits.chunk(2, dim=-1)
        else:
            lf_logits = lf_clean_logits
            layer_logits = layer_clean_logits

        # todo, mask here
        lf_probs = torch.softmax(lf_logits, dim=-1)          # B, num_leaf
        layer_probs = torch.softmax(layer_logits, dim=-1)    # B, num_layer

        if layer_skip_mask is not None:
            layer_probs = torch.masked_fill(lf_probs, layer_skip_mask, 0)

        if skip_mask is not None:
            lf_probs = torch.masked_fill(layer_probs, skip_mask, 0)

        assert sample_topk == 0

        lf_top_k_gates, lf_top_k_indices = lf_probs.topk(min(self.topk, self.num_lf_nodes), dim=-1)
        layer_top_k_gates, layer_top_k_indices = layer_probs.topk(min(self.topk, self.num_layers), dim=-1)

        lf_topk_gates, lf_mask = self.get_sparse_logits(lf_logits, lf_top_k_indices, lf_top_k_gates)
        layer_topk_gates, layer_mask = self.get_sparse_logits(layer_logits, layer_top_k_indices, layer_top_k_gates)

        return lf_topk_gates, layer_topk_gates, lf_top_k_indices, layer_top_k_indices, lf_mask, layer_mask, layer_probs


    def forward(self, lq, daclip_lq, layer_skip_mask=None, skip_mask=None):

        B, _, _, _ = lq.shape

        with torch.no_grad():
            _, degra_features = self.daclip_model.encode_image(daclip_lq, control=True)

        # estimate mu, sigma
        mu = self.mu(degra_features)
        logvar = self.sigma(degra_features)

        lf_topk_gates, layer_topk_gates, lf_topk_indices, layer_topk_indices, lf_mask, layer_mask, layer_probs = self.topk_gating(mu, logvar, layer_skip_mask=layer_skip_mask, skip_mask=skip_mask)

        converted_lf_topk_gates = layer_topk_gates[:, None, :].expand(-1, self.num_lf_nodes, -1) * lf_mask[:, :, None] + \
                                    lf_topk_gates[:, :, None].expand(-1, -1, self.num_layers) * layer_mask[:, None, :]
        converted_lf_topk_gates = converted_lf_topk_gates.sum(-1).sum(-1)  # [B, ]
        expert_indice = torch.masked_select(self.expert_indices_dict,
                                            (F.one_hot(layer_topk_indices[:, None, :].expand(-1, self.num_lf_nodes, -1).squeeze(-1), num_classes=3) * lf_mask[:, :, None]) == 1)
        lf_layer_topk_gates = torch.zeros(B, len(self.expert_pool)).to(lq.device)
        lf_layer_topk_gates.scatter_add_(1, expert_indice[:, None], converted_lf_topk_gates[:, None])

        dispatcher = SparseDispatcher(len(self.expert_pool), lf_layer_topk_gates)

        # parallel forward
        _, _, ori_H, ori_W = lq.shape
        B, _, H, W = lq.shape
        expert_inputs = dispatcher.dispatch(lq)
        gates = dispatcher.expert_to_gates()

        with torch.no_grad():
            expert_outputs = []
            for i in range(len(self.expert_pool)):
                if expert_inputs[i].shape[0] != 0:
                    out = self.expert_pool[i](expert_inputs[i])
                else:
                    out = expert_inputs[i]
                expert_outputs.append(out)
        y = dispatcher.combine(expert_outputs, gates, multiply_by_gates=self.is_scale_prob)

        cv_loss_layer = self.compute_cvloss(layer_probs) * self.cv_loss_weight
        cv_loss = cv_loss_layer
        aux_loss = cv_loss

        return y[:, :, :ori_H, :ori_W], mu, logvar, aux_loss









