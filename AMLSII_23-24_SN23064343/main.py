from A.dataset import Div2kDataset
from A.SRGAN import Generator,Discriminator
from A.loss import GeneratorLoss
from A.evaluate import evaluate,test
from A.train import train

import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import math
import torch
from torch import nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

# Data Path
ROOT = './Datasets/DIV2K_'
High_Train_ROOT = ROOT+'train_HR'
High_Val_ROOT = ROOT+'valid_HR'
High_Test_ROOT = ROOT+'test_HR'

Low_Train_bi = ROOT+'train_LR_bicubic/x4'
Low_Val_bi = ROOT+'valid_LR_bicubic/X4'
Low_Test_bi = ROOT+'test_LR_bicubic/X4'

Low_Train_un = ROOT+'train_LR_unknown/x4'
Low_Val_un = ROOT+'valid_LR_unknown/X4'
Low_Test_un = ROOT+'test_LR_unknown/X4'

# Make path to save model
out_path1 = './srgan_result/Track1'
out_path2 = './srgan_result/Track2'
if not os.path.exists(out_path1):
     os.makedirs(out_path1)
if not os.path.exists(out_path2):
     os.makedirs(out_path2)

# Check the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define hyperparams
BATCH_SIZE = 4
EPOCH = 40
lr = 0.0001

torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True

# Define Model_initialization Function
def xavier_init_weights(model):
	if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv2d):
		torch.nn.init.xavier_uniform_(model.weight)

if __name__ == 'main':
     print("======================================== Track 1 ========================================")
     train(device, High_Train_ROOT, Low_Train_bi, High_Val_ROOT, Low_Val_bi, High_Test_ROOT, Low_Test_bi, BATCH_SIZE, lr, EPOCH, out_path1)

     print("======================================== Track 2 ========================================")
     train(device, High_Train_ROOT, Low_Train_un, High_Val_ROOT, Low_Val_un, High_Test_ROOT, Low_Test_un, BATCH_SIZE, lr, EPOCH, out_path2)