import torch
import torch.nn as nn
import torch.nn.functional as F
from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer
from utils.box_utils import xywh2xyxy
import torchvision.transforms as transforms
from torchvision.transforms import Resize

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

        # TODO：这是什么意思？这个投影的目的是为了对齐视觉和语言的输出
        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        # TODO：构建融合模块
        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.device = args.device

    # 这里传入的还是原始图像和原始文本，传入的图像还是经过处理的 NestedTensor，里面包含了 图像和 mask信息
    """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels """
    def forward(self, img_data, text_data):
        # print("\n img_data shape: ", img_data.tensors.shape)  # torch.Size([32, 3, 640, 640])
        # print("\n img_data type: ", img_data.tensors.type())  # torch.cuda.FloatTensor
        # print("\n img_data mask shape: ", img_data.mask.shape)  # torch.Size([32, 640, 640])
        # print("\n img_data mask type: ", img_data.mask.type())  # torch.cuda.IntTensor
        # print("\n text_data shape: ", text_data.tensors.shape)  # torch.Size([32, 20])
        # print("\n text_data mask shape: ", text_data.mask.shape)  # torch.Size([32, 20])
        # torch_resize = Resize([256, 256], interpolation=transforms.InterpolationMode.BICUBIC)
        # torch_resize = Resize([256, 256])
        # mask_resize = torch_resize(img_data.mask)
        # print("mask_resize: ", mask_resize.shape)
        # print("mask_resize type: ", mask_resize.type())

        # 得到batch_size
        bs = img_data.tensors.shape[0]

        # visual backbone
        # resnet + pos embedding + trm encoder
        visu_mask, visu_src = self.visumodel(img_data)
        # TODO：这个投影的大小怎么确定的？里面的num_channels 在定义模型时，直接设的是 hidden_dim，所以实际形状没有变
        # print("\n visu_src shape1: ", visu_src.shape)  # torch.Size([400, 32, 256])
        # language bert
        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()
        assert text_mask is not None

        # print("\n visu_src type: ", visu_src.type())  # torch.cuda.FloatTensor
        # print("\n visu_src shape2: ", visu_src.shape)  # torch.Size([400, 32, 256])
        # print("\n visu_mask shape: ", visu_mask.shape)  # torch.Size([32, 400])
        # print("\n text_src type: ", text_src.type())  # torch.cuda.FloatTensor
        # print("\ntext_src shape: ", text_src.shape)  # torch.Size([32, 20, 768])
        # print("\ntext_mask shape: ", text_mask.shape)  # torch.Size([32, 20])，text mask 是表示有，所以需要反转

        text_src = self.text_proj(text_src)
        visu_src = self.visu_proj(visu_src)  # (N*B)xC

        # print("\n visu_src type: ", visu_src.type())  # torch.cuda.FloatTensor
        # print("\n visu_src shape2: ", visu_src.shape)  # torch.Size([400, 32, 256])
        # print("\n text_src type: ", text_src.type())  # torch.cuda.FloatTensor
        # print("\ntext_src shape: ", text_src.shape)  # torch.Size([8, 20, 256])， 文本的shape 还是对齐了

        # permute BxLenxC to LenxBxC
        # TODO: 为啥要置换？
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)  # 降成二维向量，第一维不变，其他维度合并

        # target regression token
        # tgt_src 1 * hidden_dim --> 1 x bs x hidden_dim
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        # TODO：这个 mask 到底有什么用？
        # tgt_mask bs x 1, bool, 全0
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)

        # tgt_src: 1 x bs x hidden_dim, text_src: Len x bs x hidden_dim, visu_src: N x bs x hidden_dim
        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        # tgt_mask: bs x 1, text_mask: bs x (Len x C) (二维), visu_mask: 和text_mask一样
        vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)
        # vl_pos_embed: N * hidden_dim --> N * bs * hidden_dim
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # print("vl_src size: ", vl_src.shape)  # torch.Size([421, 8, 256])
        # print("vl_src type: ", vl_src.type())  # torch.cuda.FloatTensor
        # print("vl_mask size: ", vl_mask.shape)  # torch.Size([8, 421])
        # print("vl_mask type: ", vl_mask.type())  # torch.cuda.BoolTensor
        # print("vl_pos size: ", vl_pos.shape)  # torch.Size([421, 8, 256])
        # print("vl_pos type: ", vl_pos.type())  # torch.cuda.FloatTensor

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC
        vg_hs = vg_hs[0]

        # self.bbox_embed()就是一层MLP层
        pred_box = self.bbox_embed(vg_hs).sigmoid()
        # print("\npred_box type: ", pred_box.type())  # torch.cuda.FloatTensor
        # print("\npred_box: ", pred_box)
        # pred_box: tensor([[0.4893, 0.5045, 0.5153, 0.4770],
        #                   [0.4686, 0.4845, 0.5257, 0.4959],
        #                   [0.4857, 0.5192, 0.5025, 0.5072],
        #                   [0.4984, 0.4996, 0.5099, 0.4904],
        #                   [0.5036, 0.5222, 0.5068, 0.4930],
        #                   [0.4974, 0.4884, 0.5054, 0.4827],
        #                   [0.5022, 0.5158, 0.5135, 0.4810],
        #                   [0.5051, 0.5088, 0.4947, 0.4820]], device='cuda:0', grad_fn= < SigmoidBackward0 >)
        # pred_box_tmp = torch.Tensor([[0.4893, 0.5045, 0.5153, 0.4770],
        #                              [0.4686, 0.4845, 0.5257, 0.4959],
        #                              [0.4857, 0.5192, 0.5025, 0.5072],
        #                              [0.4984, 0.4996, 0.5099, 0.4904],
        #                              [0.5036, 0.5222, 0.5068, 0.4930],
        #                              [0.4974, 0.4884, 0.5054, 0.4827],
        #                              [0.5022, 0.5158, 0.5135, 0.4810],
        #                              [0.5051, 0.5088, 0.4947, 0.4820]]).to(self.device)
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
