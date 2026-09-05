import warnings
from functools import partial

import math
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_
from torch import Tensor

import os, sys
# 把 denoising-diffusion-pytorch 目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/media/swq/8ea992ba-afb3-44ce-8a80-69ff10668360/swq/CamoDiffusion_v4/denoising-diffusion-pytorch')))
from denoising_diffusion_pytorch.simple_diffusion import ResnetBlock, LinearAttention


from model.mamba_fusion import Cross_SS2D_2, SS2D_1, SS2D_xc4

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            time_token = x[:, 0, :].reshape(B, 1, C)
            x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)  # Fixme: Check Here
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = torch.cat((time_token, x_), dim=1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, mask_chans=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if mask_chans != 0:
            self.mask_proj = nn.Conv2d(mask_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       padding=(patch_size[0] // 2, patch_size[1] // 2))
            # set mask_proj weight to 0
            self.mask_proj.weight.data.zero_()
            self.mask_proj.bias.data.zero_()  #这里的mask其实是xt

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.proj(x)
        # Do a zero conv to get the mask
        if mask is not None:
            mask = self.mask_proj(mask)
            x = x + mask
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], mask_chans=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.mask_chans = mask_chans

        # time_embed

        self.time_embed = nn.ModuleList()
        for i in range(0, len(embed_dims)):
            self.time_embed.append(nn.Sequential(
                nn.Linear(embed_dims[i], 4 * embed_dims[i]),
                nn.SiLU(),
                nn.Linear(4 * embed_dims[i], embed_dims[i]),
            ))

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0], mask_chans=mask_chans)
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward_features(self, x, timesteps, cond_img):
        time_token = self.time_embed[0](timestep_embedding(timesteps, self.embed_dims[0])) #先嵌入，在通过time_embed映射，Bxembed_dims[0]
        time_token = time_token.unsqueeze(dim=1) # Bx1xembed_dims[0]

        B = cond_img.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(cond_img, x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        time_token = x[:, 0] #提取时间步嵌入的位置（第一个位置）,[B, embed_dim(0)]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #[B, embed_dim, H, W]
        outs.append(x)

        time_token = self.time_embed[1](timestep_embedding(timesteps, self.embed_dims[1]))
        time_token = time_token.unsqueeze(dim=1) #[B, 1, embed_dims[1]]
        # stage 2
        x, H, W = self.patch_embed2(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[2](timestep_embedding(timesteps, self.embed_dims[2]))
        time_token = time_token.unsqueeze(dim=1)
        # stage 3
        x, H, W = self.patch_embed3(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        time_token = self.time_embed[3](timestep_embedding(timesteps, self.embed_dims[3]))
        time_token = time_token.unsqueeze(dim=1)

        # stage 4
        x, H, W = self.patch_embed4(x)
        x = torch.cat([time_token, x], dim=1)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        time_token = x[:, 0]
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x, timesteps, cond_img):
        x = self.forward_features(x, timesteps, cond_img)

        #        x = self.head(x[3])

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        time_token = x[:, 0, :].reshape(B, 1, C)  # Fixme: Check Here
        x = x[:, 1:, :].transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([time_token, x], dim=1)
        return x


class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b4_m(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4_m, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


from timm.models.layers import DropPath
import torch
from torch.nn import Module
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d
import torch.nn as nn


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# [B, H*W, embed_dim]
class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# [B, H*W, embed_dim]
class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# 下采样2倍 输出通道数x4
def Downsample(
        dim,
        dim_out=None,
        factor=2
):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=factor, p2=factor),
        nn.Conv2d(dim * (factor ** 2), dim if dim_out is None else dim_out, 1)
    )

# 上采样两倍 输出通道数不变
class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class Decoder(Module):
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super(Decoder, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim
        #self.fft = BlockFFT(dim=512, h=11, w=11)

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        #self.final_conv = ProgressiveFusion()
        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )
        self.time_embed1 = nn.Sequential(
            nn.Linear(self.time_embed_dim, 2 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(2 * self.time_embed_dim, 2*self.time_embed_dim),
        )
        self.time_embed2 = nn.Sequential(
            nn.Linear(2*self.time_embed_dim, 5* self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(5 * self.time_embed_dim, 5*self.time_embed_dim),
        )
        self.time_embed3 = nn.Sequential(
            nn.Linear(5*self.time_embed_dim, 8 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(8 * self.time_embed_dim, 8*self.time_embed_dim),
        )

        resnet_block = partial(ResnetBlock, groups=8)
        #下采样4倍，同时由resnet引入t
        self.down = nn.Sequential(
            ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=7, padding=3, stride=4,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )
        #新增，下采样2倍，同时由resnet引入t
        self.down_2x_1 = nn.Sequential(
            ConvModule(embedding_dim, embedding_dim*2, kernel_size=3, padding=1, stride=2, norm_cfg=dict(type='BN')),#128*44*44
            resnet_block(2*embedding_dim, 2*embedding_dim, time_emb_dim=2*self.time_embed_dim),
            ConvModule(embedding_dim*2, embedding_dim*2, kernel_size=3, padding=1, norm_cfg=dict(type='BN'))
        )

        self.down_2x_2 = nn.Sequential(
            ConvModule(embedding_dim*2, embedding_dim*5, kernel_size=3, padding=1, stride=2, norm_cfg=dict(type='BN')),
            resnet_block(5*embedding_dim, 5*embedding_dim, time_emb_dim=5*self.time_embed_dim),
            ConvModule(embedding_dim*5, embedding_dim*5, kernel_size=3, padding=1, norm_cfg=dict(type='BN'))
        )


        self.down_2x_3 = nn.Sequential(
            ConvModule(embedding_dim*5, embedding_dim*8, kernel_size=3, stride=2, padding=1, norm_cfg=dict(type='BN')),  # 3x3卷积+步长2 -> 下采样
            resnet_block(8*embedding_dim, 8*embedding_dim, time_emb_dim=8*self.time_embed_dim),
            ConvModule(embedding_dim*8, embedding_dim*8, kernel_size=3, padding=1, norm_cfg=dict(type='BN'))  # 1x1卷积调整通道数
        )
        #self.cat_conv = ConvModule(in_channels=embedding_dim * 16, out_channels=embedding_dim*8, kernel_size=1,
        #                              norm_cfg=dict(type='BN', requires_grad=True))
        # 上采样4倍 通道数变为1/8
        self.up_2x_3 = nn.Sequential(
            ConvModule(in_channels=embedding_dim*8, out_channels=embedding_dim*5, kernel_size=1,
                       norm_cfg=dict(type='BN')),
            Upsample(embedding_dim*5, embedding_dim*5, factor=2),
            ConvModule(in_channels=embedding_dim*5, out_channels=embedding_dim*5, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN')),
        )
        self.up_2x_2 = nn.Sequential(
            ConvModule(in_channels=embedding_dim*5, out_channels=embedding_dim*2, kernel_size=1,
                       norm_cfg=dict(type='BN')),
            Upsample(embedding_dim*2, embedding_dim*2, factor=2),
            ConvModule(in_channels=embedding_dim*2, out_channels=embedding_dim*2, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN')),
        )
        self.up_2x_1 = nn.Sequential(
            ConvModule(in_channels=embedding_dim*2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN')),
            Upsample(embedding_dim, embedding_dim, factor=2),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN')),
        )

        self.up = nn.Sequential(
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            # resnet_block(embedding_dim, embedding_dim),
            Upsample(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )

        self.pred = nn.Sequential(
            # ConvModule(in_channels=embedding_dim//8+1, out_channels=embedding_dim//8, kernel_size=1,
            #            norm_cfg=dict(type='BN', requires_grad=True)),
            nn.Dropout(0.1),
            Conv2d(embedding_dim // 8, self.num_classes, kernel_size=1)
        )
        
        self.bn1 = nn.BatchNorm2d(embedding_dim)
        self.bn2 = nn.BatchNorm2d(embedding_dim * 2)
        self.bn3 = nn.BatchNorm2d(embedding_dim * 5)
        self.bn4 = nn.BatchNorm2d(embedding_dim * 8)
        
        self.ssm1 = SS2D_xc4(embedding_dim)
        self.ssm2 = SS2D_xc4(embedding_dim*2)
        self.ssm3 = SS2D_xc4(embedding_dim*5)
        self.ssm4 = SS2D_xc4(embedding_dim*8)
        

    
    def forward(self, inputs, timesteps, x):
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
        c1, c2, c3, c4 = inputs

        ##############################################下采样，融合时间信息
        _x = [x]
        for blk in self.down:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)

        xc1 = self.ssm1(x.permute(0, 2, 3, 1),c1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = xc1

        t1 = self.time_embed1(t)
        for blk in self.down_2x_1:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t1)
                _x.append(x)
            else:
                x = blk(x)

        xc2 = self.ssm2(x.permute(0, 2, 3, 1),c2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = xc2

        t2 = self.time_embed2(t1)
        for blk in self.down_2x_2:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t2)
                _x.append(x)
            else:
                x = blk(x)

        xc3 = self.ssm3(x.permute(0, 2, 3, 1),c3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = xc3

        t3 = self.time_embed3(t2)

        for blk in self.down_2x_3:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t3)
                _x.append(x)
            else:
                x = blk(x) 

        xc4 = self.ssm4(x.permute(0, 2, 3, 1),c4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = xc4
       ####################### up_sample ########################


        for blk in self.up_2x_3:   #上采样没有加入t
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)

        x = x + xc3
        for blk in self.up_2x_2:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)

        x = x + xc2
        for blk in self.up_2x_1:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)
        
        x = x + xc1
        for blk in self.up:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)

        x = self.pred(x)
        return x, c1, c2, c3, c4


class net(nn.Module):
    def __init__(self, class_num=2, mask_chans=0, **kwargs):
        super(net, self).__init__()
        self.class_num = class_num
        self.backbone = pvt_v2_b4_m(in_chans=3, mask_chans=mask_chans)
        self.depth_backbone = pvt_v2_b4_m(in_chans=1, mask_chans=mask_chans)
        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=64, class_num=class_num, mask_chans=mask_chans)
        self._init_weights()  # load pretrain

        self.frequency_based_fusion = nn.ModuleList([
            FrequencyAttention(dim=64, img_size=88),
            FrequencyAttention(dim=128, img_size=44),
            FrequencyAttention(dim=320, img_size=22),
            FrequencyAttention(dim=512, img_size=11)
        ])
         
    def forward(self, x, timesteps, cond_img, depth_map):
        rgb_features = self.backbone(x, timesteps, cond_img)
        depth_features = self.depth_backbone(None, timesteps, depth_map)

        features = []
        for i, (rgb_feat, depth_feat) in enumerate(zip(rgb_features, depth_features)):
            # 执行特征融合
            
            fused = self.frequency_based_fusion[i](rgb_feat, depth_feat)
            features.append(fused)
            
        features, _, _, _, _ = self.decode_head(features, timesteps, x)
        return features

    def _download_weights(self, model_name):
        _available_weights = [
            'pvt_v2_b0',
            'pvt_v2_b1',
            'pvt_v2_b2',
            'pvt_v2_b3',
            'pvt_v2_b4',
            'pvt_v2_b4_m',
            'pvt_v2_b5',
        ]
        assert model_name in _available_weights, f'{model_name} is not available now!'
        from huggingface_hub import hf_hub_download
        return hf_hub_download('Anonymity/pvt_pretrained', f'{model_name}.pth', cache_dir='./pretrained_weights')

    def _init_weights(self):
        pretrained_dict = torch.load('./pretrained_weights/pvt_v2_b4_m.pth', weights_only=True) #for save mem
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=False)

    @torch.inference_mode()
    def sample_unet(self, x, timesteps, cond_img, depth_map):
        return self.forward(x, timesteps, cond_img, depth_map)

    def extract_features(self, cond_img, depth_map):
        # do nothing
        return cond_img, depth_map


class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass

    
#rgb-d fusion based frequency
class FrequencyAttention(nn.Module):
    def __init__(self, dim, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=96,norm_layer=nn.LayerNorm, ape=False, patch_norm=True, drop_path: float = 0):
        super().__init__()

        self.conv_amp1 = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU())
        self.conv_amp2 = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU())

        self.conv_pha1 = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU())    
        self.conv_pha2 = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU())

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
     
        self.final_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        

        self.ssm1 = SS2D_1(dim)
        self.ssm2 = SS2D_1(dim)
        self.ssm = Cross_SS2D_2(dim, d_state="auto")#SS2D_3(dim)


        #self.drop_path = DropPath(drop_path)

    def forward(self, f_rgb, f_dep):
        
        _, _, h, w = f_rgb.shape
        fre_rgb = torch.fft.rfft2(f_rgb)
        fre_dep = torch.fft.rfft2(f_dep)


        amp_rgb = torch.abs(fre_rgb)
        pha_rgb = torch.angle(fre_rgb)
        amp_dep = torch.abs(fre_dep)
        pha_dep = torch.angle(fre_dep)
        
        amp_rgb = self.conv_amp1(amp_rgb)
        amp_dep = self.conv_amp2(amp_dep)
        pha_rgb = self.conv_pha1(pha_rgb)
        pha_dep = self.conv_pha2(pha_dep)
        
        x1 = torch.fft.irfft2(amp_rgb * torch.exp(1j * pha_dep), s=(h, w))
        x2 = torch.fft.irfft2(amp_dep * torch.exp(1j * pha_rgb), s=(h, w))
        """
        x_ = self.final_conv(x1+x2)

        """
        x1 = self.ssm1(x1.permute(0, 2, 3, 1))#b,h,w,c
        x2 = self.ssm2(x2.permute(0, 2, 3, 1))

        x_ = self.ssm(x1, x2).permute(0, 3, 1, 2) # b,c,h,w
       
        return x_
   