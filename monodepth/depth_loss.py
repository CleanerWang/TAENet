from __future__ import absolute_import, division, print_function
import torch.nn as nn
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms, datasets
from . import networks
from .layers import disp_to_depth
from .utils import download_model_if_doesnt_exist
import sys
feed_height = 192
feed_width = 640
sys.setrecursionlimit(1000000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_path = "monodepth/models/mono+stereo_640x192/encoder.pth"
depth_decoder_path =  "monodepth/models/mono+stereo_640x192/depth.pth"
encoder = networks.ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)

# extract the height and width of image that this model was trained with
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
for param in encoder.parameters():
    param.requires_grad = False
encoder.to(device)
encoder.eval()

depth_decoder = networks.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)
for param in depth_decoder.parameters():
    param.requires_grad = False
depth_decoder.to(device)
depth_decoder.eval()
class  depth_loss(torch.nn.Module):
    def __init__(self, ablation=False):
        super(depth_loss, self).__init__()
        pass

    def forward(self, inputs, labels):
        feature1 = encoder(inputs)
        output1 = depth_decoder(feature1)[("disp", 0)]

        feature2 = encoder(labels)
        output2 = depth_decoder(feature2)[("disp", 0)]

        criterion = nn.MSELoss().to(device)
        loss = criterion(output1,output2)
        return loss