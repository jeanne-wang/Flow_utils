import numpy as np
import os.path
from skimage.io import imread, imsave
from scipy.io import loadmat
import torch
from torch.autograd import Variable
import torch.nn as nn

device = torch.device("cuda:3")
torch.cuda.set_device(device)

flow_file='../siftflow_examples/NBA2K19_2019.01.31_23.50.52_frame1001579_players_noised_camera_5.siftflow.mat'
image_with_noised_camera_file='/projects/grail/xiaojwan/2k_players_mesh_rasterized_noised_camera_sigma_5/NBA2K19_2019.01.31_23.50.52_frame1001579.png'
image_file='/projects/grail/xiaojwan/nba2k_flow/2k_frames/NBA2K19_2019.01.31_23.50.52_frame1001579.png'
output_prefix='NBA2K19_2019.01.31_23.50.52_frame1001579.siftflow'
image_with_noised_camera = imread(image_with_noised_camera_file)

if flow_file[-3:]='.mat':
	flow = loadmat(flow_file)['flow']
else:
	flow=np.load(flow_file)
image=imread(image_file)[...,:3]
flow_zero_mask = np.all((flow==0), axis=2, keepdims=True)
flow_zero_mask = np.expand_dims(flow_zero_mask, axis=0)

image_with_noised_camera = np.expand_dims(image_with_noised_camera, axis=0)
image_with_noised_camera = image_with_noised_camera.transpose(0, 3, 1, 2)
image_with_noised_camera = torch.from_numpy(image_with_noised_camera).float().to(device)

B, C, H, W = image_with_noised_camera.size() ## B=1
	
# mesh grid 
xx = torch.arange(0, W).view(1,-1).repeat(H,1)
yy = torch.arange(0, H).view(-1,1).repeat(1,W)
xx = xx.view(1,1,H,W).repeat(B,1,1,1)
yy = yy.view(1,1,H,W).repeat(B,1,1,1)
grid = torch.cat((xx,yy),1).float().to(device)

flow = np.expand_dims(flow, axis=0)
flow = flow.transpose(0, 3, 1, 2)
flow = torch.from_numpy(flow).float().to(device)
vgrid = Variable(grid)+flow

# scale grid to [-1,1] 
vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

vgrid = vgrid.permute(0,2,3,1)        
output = nn.functional.grid_sample(image_with_noised_camera, vgrid)



output = output.cpu().numpy()
output = output.transpose(0,2,3,1)

# for flow value = 0, use original pxiel value
output = output * (1-flow_zero_mask) + image * flow_zero_mask
imsave(output_prefix+'.warp.png', output[0])

