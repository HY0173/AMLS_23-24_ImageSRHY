import os
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

class Div2kDataset(Dataset):
    '''Get images under path of data_dir into list'''
    def __init__(self, data_dir, transform=ToTensor()):
        self.file_paths = sorted([os.path.join(data_dir, x) for x in os.listdir(data_dir) if is_image_file(x)])
        self.transform = transform

    def __getitem__(self, index):
         # file_name = self.file_paths[index].split('/')[-1]
         img = Image.open(self.file_paths[index])
         img = self.transform(img)
         return img
    
    def __len__(self):
        return len(self.file_paths)
    