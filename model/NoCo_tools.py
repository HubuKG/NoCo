from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math

from .Tool_model import GAT, GCN
# from model.Tool_model import GAT,GCN
import pdb

class NoiseAugment(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers, modalities):
        super(NoiseAugment, self).__init__()
        self.emb_size = emb_size
        self.modalities = modalities
        self.model = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ) for _ in range(num_layers - 2)],
            nn.Linear(hidden_size, emb_size)
        )

        self.noise_scales = {modality: 1.0 for modality in modalities}

    def forward(self, embeddings, num_steps):
        for modality, emb in embeddings.items():
            input_size = emb.shape[-1]
            noise_scale = self.noise_scales[modality]

            for _ in range(num_steps):
                noise = torch.randn_like(emb) * noise_scale
                noisy_emb = emb + noise

                transform_to_fixed = nn.Linear(input_size, self.emb_size).cuda()
                noisy_emb = transform_to_fixed(noisy_emb)

                denoised_emb = self.model(noisy_emb)

                transform_to_original = nn.Linear(self.emb_size, input_size).cuda()
                denoised_emb = transform_to_original(denoised_emb)

                noise_scale = self.update_noise_scale(modality, noise_scale, emb, denoised_emb)

                emb = denoised_emb

            embeddings[modality] = denoised_emb

        return embeddings

    def update_noise_scale(self, modality, noise_scale, original_emb, denoised_emb):
        if modality == 'text':
            similarity = torch.cosine_similarity(original_emb, denoised_emb, dim=-1).mean()
            if similarity > 0.9:
                noise_scale *= 0.9
            else:
                noise_scale *= 1.1

        elif modality == 'image':
            variance = torch.var(denoised_emb)
            if variance < 0.1:
                noise_scale *= 1.2
            else:
                noise_scale *= 0.8

        elif modality == 'relation':
            distance = torch.norm(original_emb - denoised_emb, dim=-1).mean()
            if distance > 0.5:
                noise_scale *= 0.85
            else:
                noise_scale *= 1.05

        return noise_scale

class NoCoFusion(nn.Module):
    def __init__(self, args, modal_num, ent_num, with_weight=1):
        super().__init__()
        self.args = args
        self.modal_num = modal_num
        self.ENT_NUM = ent_num

        self.mono_confidence = nn.ModuleList([nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, 1),
            nn.Sigmoid()
        ) for _ in range(modal_num)])

        self.holo_confidence = nn.Sequential(
            nn.Linear(4, self.ENT_NUM),
            nn.ReLU(),
            nn.Linear(self.ENT_NUM, 4),
            nn.Sigmoid()
        )

    def calculate_relative_calibration(self, embs):
        uncertainties = []
        for emb in embs:
            probs = F.softmax(emb, dim=-1)  # (batch_size, hidden_size)
            uniformity = torch.mean(torch.abs(probs - probs.mean(dim=-1, keepdim=True)), dim=-1)  # (batch_size,)
            uncertainties.append(uniformity)

        calibration_factors = []
        for i, uncertainty in enumerate(uncertainties):
            relative_uncertainty = uncertainty / (sum(uncertainties) - uncertainty + 1e-8)
            calibration_factors.append(relative_uncertainty.unsqueeze(-1))  # (batch_size, 1)

        return torch.cat(calibration_factors, dim=-1)  # (batch_size, modal_num)

    def forward(self, embs):
        embs = [embs[idx] for idx in range(len(embs)) if embs[idx] is not None]
        modal_num = len(embs)

        hidden_states = torch.stack(embs, dim=1)  # (batch_size, modal_num, hidden_size)
        bs = hidden_states.shape[0]

        mono_confidences = []
        for idx in range(modal_num):
            mono_conf = self.mono_confidence[idx](embs[idx])
            mono_confidences.append(mono_conf)

        mono_confidences = torch.cat(mono_confidences, dim=-1)  # (batch_size, modal_num)

        holo_confidences = self.holo_confidence(mono_confidences)  # (batch_size, modal_num)

        fusion_weights = mono_confidences + holo_confidences  # element-wise乘法 (batch_size, modal_num)

        calibration_factors = self.calculate_relative_calibration(hidden_states)  # (batch_size, modal_num)

        calibrated_weights = fusion_weights * calibration_factors  # (batch_size, modal_num)

        fusion_weights = F.softmax(calibrated_weights, dim=-1)  # (batch_size, modal_num)

        weighted_embs = [fusion_weights[:, idx].unsqueeze(1) * F.normalize(embs[idx]) for idx in range(modal_num)]

        joint_emb = torch.cat(weighted_embs, dim=1)  # (batch_size, modal_num, hidden_size)

        return joint_emb, hidden_states, fusion_weights


class MultiModalEncoder(nn.Module):

    def __init__(self, args,
                 ent_num,
                 img_feature_dim,
                 char_feature_dim=None,
                 use_project_head=False,
                 attr_input_dim=1000):
        super(MultiModalEncoder, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        name_dim = self.args.name_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])

        #########################
        # Entity Embedding
        #########################
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)  # 创建实体嵌入层
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))  # 初始化嵌入层权重
        self.entity_emb.requires_grad = True  # 设置权重为可训练参数

        #########################
        # Modal Encoder
        #########################
        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(attr_input_dim, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)

        # structure encoder
        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)

        #########################
        # Fusion Encoder
        #########################
        self.fusion = NoCoFusion(args, modal_num=self.args.inner_view_num,
                                    ent_num=self.ENT_NUM,with_weight=self.args.with_weight)

        self.noiseupdate = NoiseAugment(64,64,4,['text','image','relation'])

    def forward(self,
                input_idx,  # 实体索引
                adj,  # 邻接矩阵
                img_features=None,
                rel_features=None,
                att_features=None,
                name_features=None,
                char_features=None):

        # 图邻域结构嵌入
        if self.args.w_gcn:
            gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
        else:
            gph_emb = None
        # 图像嵌入
        if self.args.w_img:
            img_emb = self.img_fc(img_features)
        else:
            img_emb = None
        # 关系嵌入
        if self.args.w_rel:
            rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None
        # 属性嵌入
        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None
        if self.args.w_name and name_features is not None:
            name_emb = self.name_fc(name_features)
        else:
            name_emb = None
        if self.args.w_char and char_features is not None:
            char_emb = self.char_fc(char_features)
        else:
            char_emb = None

        embeddings = {
            'text':att_emb,
            'image':img_emb,
            'relation':rel_emb
        }

        denoised_embeddings = self.noiseupdate(embeddings,2)
        img_emb = denoised_embeddings['image']
        att_emb = denoised_embeddings['text']
        rel_emb = denoised_embeddings['relation']

        joint_emb, hidden_states, weight_norm = self.fusion([img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb])


        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb, hidden_states, weight_norm