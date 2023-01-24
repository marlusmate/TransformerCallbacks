import torch
import torch.utils.checkpoint as checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch import roll, cat
from timm.models.layers import trunc_normal_, DropPath
import numpy as np

def window_partition2D(x, window_size):
    """
    from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse2D(windows, window_size, H, W):
    """
    from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def calc_sw_msa_mask(input_res, window_size, shift_size):
    H, W = input_res
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, window_size[0]),
                        slice(window_size[0], shift_size[0]),
                        slice(shift_size[0], None))
    w_slices = (slice(0, window_size[1]),
                        slice(window_size[1], shift_size[1]),
                        slice(shift_size[1], None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    return img_mask

def create_relative_cords_table2D(window_size, pretrained_window_size=[0,0]):
            # get relative_coords_table
    relative_coords_h = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32)
    relative_coords_w = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32)
    relative_coords_table = torch.stack(
        torch.meshgrid([relative_coords_h,
                        relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
    if pretrained_window_size > 0:
        relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
    else:
        relative_coords_table[:, :, :, 0] /= (window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (window_size[1] - 1)
    relative_coords_table *= 8  # normalize to -8, 8
    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / np.log2(8)
    return relative_coords_table


def get_pairwise_relative_pos_idx2D(window_size):
# get pair-wise relative position index for each token inside the window
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

class FeedForwardNetwork(nn.Module):
    """
    from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
    2 Dense Layer FFN
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop1=0., drop2=0., actv=nn.GELU) -> None:
        super(FeedForwardNetwork, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, int(hidden_features))
        self.act = actv()
        self.fc2 = nn.Linear(int(hidden_features), out_features)
        self.drop1 = nn.Dropout(drop1)
        self.drop2 = nn.Dropout(drop2)
        
    
    def forward(self, x):
        """
        Input / Output Shape: [Batch, n_tokens, embed_dim]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, pool_stride:tuple=(2,2),norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        n_tk_in = pool_stride[0]*pool_stride[1]
        n_tk_out = n_tk_in // 2
        self.reduction = nn.Linear(n_tk_in * dim, n_tk_out * dim, bias=False)
        #self.const = 
        self.norm = norm_layer(2 * dim)
        self.pool_stride = pool_stride

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        #assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        """
        if H % self.pool_stride[0] !=0:
            stride_h = None
            num_steps_h = [0]
        else:
            stride_h = self.pool_stride[0]
            num_steps_h = range(H//self.pool_stride[0])

        if W % self.pool_stride[1] !=0:
            stride_w = None
            num_steps_w = [0]
        else:
            stride_w = self.pool_stride[1]
            num_steps_w = range(W//self.pool_stride[1])

        pool_parts = []
        for pool_w in num_steps_w:
            for pool_h in num_steps_h:
                pool_parts.append(x[:, pool_h::stride_h, pool_w::stride_w,:])
        """
        x0 = x[:, 0::2, 0::2, :]  # B H/stride_h W/stride_w C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        
        #xt = cat(pool_parts, -1)
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

class MultiheadCosineAttention2D(nn.Module):
    """
    snipped from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
    """
    def __init__(self, dim, num_heads, pe='None', window_size=[1,1], pretrained_window_size=[0,0], qkv_bias=True, attn_drop=0., proj_drop=0.) -> None:
        super(MultiheadCosineAttention2D, self).__init__()

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        if pe == 'relative':
            self.relative_pos_bias = RelativePositionBias2D(dim=dim, window_size=window_size, num_heads=num_heads, pretrained_window_size=pretrained_window_size)
        else:
            self.relative_pos_bias = None

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)


        self.max = torch.log(torch.tensor(1. / 0.01)).to("cuda")
    def forward(self, x, mask=None):
        B_, N, C = x.shape

        # split into heads
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=self.max).exp()
        attn = attn * logit_scale

        # optional add relative position bias
        if self.relative_pos_bias is not None:
            attn = attn + self.relative_pos_bias()

        # optional masking
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RelativePositionBias2D(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, pretrained_window_size=[0, 0]):
        super(RelativePositionBias2D, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_table = create_relative_cords_table2D(window_size=window_size, pretrained_window_size=pretrained_window_size)
        self.register_buffer("relative_coords_table", relative_coords_table)

        relative_position_index = get_pairwise_relative_pos_idx2D(self.window_size)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        
        return relative_position_bias.unsqueeze(0)

class SwinTFV2Block2D(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, 
            dim, 
            input_resolution, 
            num_heads, 
            window_size=(7,7), 
            shift_size=(0,0),
            mlp_ratio=4., 
            qkv_bias=True, 
            drop=0., 
            attn_drop=0., 
            drop_path=0.,
            act_layer=nn.GELU, 
            norm_layer=nn.LayerNorm, 
            pretrained_window_size=(0,0), 
            pe='relative'
        ):
        super(SwinTFV2Block2D, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= max(self.window_size):
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = (0,0)
            self.window_size = (min(self.input_resolution), min(self.input_resolution))
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = MultiheadCosineAttention2D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=pretrained_window_size, pe=pe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForwardNetwork(in_features=dim, hidden_features=mlp_hidden_dim, actv=act_layer, drop1=drop, drop2=drop)

        if any(val > 0 for val in self.shift_size):
            # calculate attention mask for SW-MSA
            img_mask = calc_sw_msa_mask(self.input_resolution, self.window_size, self.shift_size)
            mask_windows = window_partition2D(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if any(val > 0 for val in self.shift_size):
            shifted_x = roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition2D(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse2D(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if any(val > 0 for val in self.shift_size):
            x = roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class SwinTFV2BasicLayer2D(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, pool_stride=(2,2),use_checkpoint=False,
                 pretrained_window_size=0, pe='relative'):

        super(SwinTFV2BasicLayer2D, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTFV2Block2D(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0,0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2),
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size,
                                 pe = pe)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, pool_stride=pool_stride, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, 
        img_size=(224,224),
        patch_size=(4,4), 
        in_chans=1, 
        embed_dim=96, 
        norm_layer=None,
        use_pretrained=False,
        weights_dir=None):
        super(PatchEmbed2D, self).__init__()
        img_size = img_size
        patch_size = patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        if use_pretrained:
            embed_dict = torch.load(weights_dir+'embedding.yaml')
            layers_dict = torch.load(weights_dir+'layers.yaml')
            self.proj.weight = embed_dict['embed_weight']
            self.proj.bias = embed_dict['embed_bias']

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(
            self, 
            img_size=(224,224), 
            patch_size=(4,4), 
            in_chans=1, 
            num_classes=3,
            embed_dim=96, 
            depths=[2, 2, 6, 2], 
            num_heads=[3, 6, 12, 24],
            window_size=(7,7),
            pools_stride=(2,2), 
            mlp_ratio=4., 
            qkv_bias=True,
            drop_rate=0., 
            attn_drop_rate=0., 
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, 
            pe='relative', 
            patch_norm=True,
            use_checkpoint=False, 
            pretrained_window_sizes=[0, 0, 0, 0], 
            pred=True, 
            load_pretrained = '',
            pretrained="Dictionaries/swinv2-tiny-patch4-window7-224",
            **kwargs
        ):
        super(SwinTransformerV2, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.pe = pe
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.prerained = pretrained

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.pe == 'absolute':
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        input_res_h = [patches_resolution[0]]
        input_res_w = [patches_resolution[1]]
        for _ in range(self.num_layers):
            if input_res_h[-1] % pools_stride[0] == 0:
                input_res_h.append(input_res_h[-1] // pools_stride[0])
            else:
                input_res_h.append(input_res_h[-1])
            if input_res_w[-1] % pools_stride[1] == 0:
                input_res_w.append(input_res_w[-1] // pools_stride[1])
            else:
                input_res_w.append(input_res_h[-1])

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinTFV2BasicLayer2D(dim=int(embed_dim * 2 ** i_layer),
                               #input_resolution=(patches_resolution[0] // (pools_stride[0] ** i_layer),
                                                 #patches_resolution[1] // (pools_stride[1] ** i_layer)),
                               input_resolution=(input_res_h[i_layer], input_res_w[i_layer]),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                               pool_stride=pools_stride,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if pred:
            self.head = nn.Sequential(nn.Linear((input_res_h[i_layer], input_res_w[i_layer]), 3),
            nn.Softmax(dim=-1))
        else: self.head = None

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        if load_pretrained != 'skip' : 
            self.load_pretrained()

    def load_pretrained(self):
        state_dict = torch.load(self.pretrained, map_location='cpu')

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]
        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight']#.unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]
        self.load_state_dict(state_dict, strict=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.pe == 'absolute':
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.head is not None: 
            x = self.avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
            x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
