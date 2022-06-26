import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel
from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer
from utils.box_utils import xywh2xyxy


class TransVG(nn.Module):
    def __init__(self, args):
        super(TransVG, self).__init__()
        # args.vl_hidden_dim 默认 256
        hidden_dim = args.vl_hidden_dim
        # 默认没有 dilation，是 32，用于最后一层卷积层
        divisor = 16 if args.dilation else 32
        # 一幅图像 预处理后输入是 640*640，640 / 32 = 20, 一个 patch 32*32，视觉token 是 20*20
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        # 文本token即是最长的查询长度, 默认20
        self.num_text_token = args.max_query_len
        # TODO：视觉模态用 detr 处理，构建的 DETR 包含 resnet 作为 backbone，加上 pos embedding，再加上仅 encoder的 Transformer
        self.visumodel = build_detr(args)
        # TODO：语言模态用 bert 处理，BertModel.from_pretrained(name)，直接用库进行加载，同时设置参数免更新
        self.textmodel = build_bert(args)

        # TODO: 这个是融合层的token总数，视觉 token + 文本 token + [REG] token?
        num_total = self.num_visu_token + self.num_text_token + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        # TODO：这是什么意思？
        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        # TODO：构建融合模块
        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    # 这里传入的还是原始图像和原始文本
    def forward(self, img_data, text_data):
        # 得到batch_size
        bs = img_data.tensors.shape[0]

        # visual backbone
        # resnet + pos embedding + trm encoder
        visu_mask, visu_src = self.visumodel(img_data)
        # TODO：这个投影的大小怎么确定的？
        visu_src = self.visu_proj(visu_src)  # (N*B)xC

        # language bert
        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()
        assert text_mask is not None
        text_src = self.text_proj(text_src)
        # permute BxLenxC to LenxBxC
        # TODO: 为啥要置换？
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)

        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        # TODO：这个 mask 到底有什么用？
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)
        
        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC
        vg_hs = vg_hs[0]

        # self.bbox_embed()就是一层MLP层
        pred_box = self.bbox_embed(vg_hs).sigmoid()

        return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
