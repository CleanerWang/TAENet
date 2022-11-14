import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
from Networks.Ori_ours import dehaze_net as net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def dehaze_image(image_path):

	data_hazy = Image.open(image_path)
	data_hazy = data_hazy.resize((512, 256), Image.ANTIALIAS)
	data_hazy = (np.asarray(data_hazy)/255.0)

	data_hazy = torch.from_numpy(data_hazy).float()
	data_hazy = data_hazy.permute(2,0,1)
	data_hazy = data_hazy.to(device).unsqueeze(0)
	#data_hazy = data_hazy.unsqueeze(0)
	dehaze_net = torch.nn.DataParallel(net().to(device))
	#dehaze_net = net().to(device)
	dehaze_net.load_state_dict(torch.load('snapshots/GCA_All_in_One_no_trans/Epoch99.pth',map_location=device))

	clean_image = dehaze_net(data_hazy)
	#img = torch.cat((data_hazy, clean_image),0)
	torchvision.utils.save_image( clean_image,"/data/fwc2/code/GCA1/result/" + image_path.split("/")[-1])
	torch.cat((data_hazy, clean_image),0)

if __name__ == '__main__':

	test_list = glob.glob("/data/fwc2/datasets/datasets/Test_data/dusty/trainA/*")

	for image in test_list:

		dehaze_image(image)
		print(image, "done!")
