import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import *

class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)



class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x+y)


class dehaze_net(nn.Module):
    def __init__(self, in_c=3, out_c=3, only_residual=True,in_chans=3, out_chans=3, window_size=8,
                 embed_dims=[32, 64, 64, 64, 32],
				 mlp_ratios=[2., 4., 4., 2., 2.],
				 depths=[16, 16, 16, 8, 8],
				 num_heads=[2, 4, 6, 1, 1],
				 attn_ratio=[1/4, 1/2, 3/4, 0, 0],
				 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
				 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super(dehaze_net, self).__init__()

        self.res1 = ResidualBlock(64, dilation=2)
        self.res2 = ResidualBlock(64, dilation=2)
        self.res3 = ResidualBlock(64, dilation=2)
        self.res4 = ResidualBlock(64, dilation=4)
        self.res5 = ResidualBlock(64, dilation=4)
        self.res6 = ResidualBlock(64, dilation=4)
        self.res7 = ResidualBlock(64, dilation=1)

        self.deconv1 = nn.Conv2d(64, out_c, 1)
        self.only_residual = only_residual

        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone

        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])



    def forward(self, x):
        x = self.patch_embed(x)  # 4x24x256x256
        x = self.layer1(x)  # 4X24X256X256

        y1 = self.patch_merge1(x)  #4X48X128X128
        y1 = self.layer2(y1)

        y = self.res1(y1)
        y = self.res2(y)
        y = self.res3(y)
        y2 = self.res4(y)
        y = self.res5(y2)
        y = self.res6(y)
        y3 = self.res7(y)

        x = self.patch_split1(y3)
        if self.only_residual:
            y = self.deconv1(x)
        else:
            y = F.relu(self.deconv1(x))

        return y
