import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
#from skimage.measure import compare_psnr
#from skimage.measure import compare_ssim
import dataloader
from Networks.AIENet import dehaze_net as net
from metrics import get_psnr_ssim
import numpy as np
from torchvision import transforms
from CR import  ContrastLoss
from monodepth.depth_loss import depth_loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth_loss = depth_loss().to(device)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

	dehaze_net = nn.DataParallel(net().to(device))

	train_dataset = dataloader.dehazing_loader(config.orig_images_path,
											   config.hazy_images_path)
	val_dataset = dataloader.dehazing_loader(config.orig_images_path,
											 config.hazy_images_path, mode="val")
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

	criterion = nn.MSELoss().to(device)
	optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

	dehaze_net.train()
	file = open(r"metric.txt", "a")

	for epoch in range(config.num_epochs):
		print("start train")
		for iteration, (img_orig, img_haze) in enumerate(train_loader):

			img_orig = img_orig.to(device)
			img_haze = img_haze.to(device)

			clean_image = dehaze_net(img_haze)

			loss1 = criterion(clean_image, img_orig)
			contrast_loss = ContrastLoss(ablation=False)
			l1_loss = nn.L1Loss().to(device)
			#
			loss_l1 = l1_loss(clean_image, img_orig)
			loss_contrast = 0.1*contrast_loss(clean_image,img_orig, img_haze)
			loss_depth = 10*depth_loss(clean_image, img_orig)
			#
			


			loss =loss_l1   + loss_depth #+loss_contrast

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(dehaze_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("[epoch:%2d/%d][train_iteration: %2d][loss:%.7f][l1_loss:%.7f][depth_loss:%.7f][contrast_loss:%.7f]" % (1 + epoch, config.num_epochs,iteration, loss.item(),loss_l1,loss_depth,loss_contrast))
				#print("[epoch:%2d/%d][train_iteration: %2d][loss:%.7f]]" % (1 + epoch, config.num_epochs,iteration,loss))
			if ((iteration+1) % config.snapshot_iter) == 0:
				
				torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')



		#Validation Stage
		print("start validation")
		psnrs = []
		ssims = []
		for iter_val, (img_orig, img_haze) in enumerate(val_loader):


			img_orig = img_orig.to(device)

			img_haze = img_haze.to(device)

			clean_image = dehaze_net(img_haze)
			psnr,ssim = get_psnr_ssim(img_orig, clean_image)
			psnrs.append(psnr)
			ssims.append(ssim)

			print("[epoch:%2d/%d][val_image:%2d/][PSNR : %.7f SSIM : %.7f ]" % (1 + epoch, config.num_epochs,iter_val, psnr,ssim))


			torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig),0), config.sample_output_folder+str(iter_val+1)+".jpg")
		Average_psnr = sum(psnrs) / len(psnrs)
		Average_ssim = sum(ssims) / len(ssims)
		print("[epoch:%2d/%d][Average_psnr: %.7f  Average_ssim:%.7f]" % (1 + epoch, config.num_epochs,Average_psnr,Average_ssim))
		file = open(r"metric.txt", "a")
		file.writelines(["\nEpoch: ", str(epoch), "   Average_psnr: ", str(Average_psnr),  "  Average_psnr: ", str(Average_ssim), "  loss ", str(float(loss)),"  depth_loss ", str(float(loss_depth))])


		torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")
	file.close()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--orig_images_path', type=str, default=r'C:/Users/fwc/Desktop/datasets/Train_data/dusty/label/')
	parser.add_argument('--hazy_images_path', type=str, default=r'C:/Users/fwc/Desktop/datasets/Train_data/dusty/dusty/')
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--train_batch_size', type=int, default=1)
	parser.add_argument('--val_batch_size', type=int, default=1)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--sample_output_folder', type=str, default="samples/")


	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)
	if not os.path.exists(config.sample_output_folder):
		os.mkdir(config.sample_output_folder)




	train(config)






