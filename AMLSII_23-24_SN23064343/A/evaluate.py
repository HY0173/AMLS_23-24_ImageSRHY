import numpy as np
from math import log10
import torch
from piq import ssim, psnr
import sys, os
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor
from glob import glob
from piq.utils import _validate_input, _reduce
from piq.functional import gaussian_filter
from typing import Tuple, Union, List
from piq.ssim import _ssim_per_channel_complex,_ssim_per_channel

def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
         data_range: Union[int, float] = 1., reduction: str = 'mean', full: bool = False,
         downsample: bool = True, k1: float = 0.01, k2: float = 0.03) -> List[torch.Tensor]:
    x = x / float(data_range)
    y = y / float(data_range)

    # Averagepool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if (f > 1) and downsample:
        x = nn.functional.avg_pool2d(x, kernel_size=f)
        y = nn.functional.avg_pool2d(y, kernel_size=f)

    kernel = gaussian_filter(kernel_size, kernel_sigma, device=x.device, dtype=x.dtype).repeat(x.size(1), 1, 1, 1)
    _compute_ssim_per_channel = _ssim_per_channel_complex if x.dim() == 5 else _ssim_per_channel
    ssim_map, cs_map = _compute_ssim_per_channel(x=x, y=y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    ssim_val = _reduce(ssim_val, reduction)
    cs = _reduce(cs, reduction)

    if full:
        return [ssim_val, cs]

    del x,y,f,kernel,_compute_ssim_per_channel,ssim_map, cs_map,cs,
    torch.cuda.empty_cache()

    return ssim_val

def load_img_to_tensor(filepath):
	img = Image.open(filepath)
	return ToTensor()(img)

def test(D,G,hr_loader,lr_loader):
     G.eval()
     D.eval()
     with torch.no_grad():
        test_psnr_result = torch.tensor([]).cuda()
        test_ssim_result = torch.tensor([]).cuda()
        for (batch, hr_batch), lr_batch in tqdm(zip(enumerate(hr_loader), lr_loader),total=len(lr_loader)):
            #hr_img, lr_img = hr_batch.cuda(), lr_batch.cuda()
            hr_img, lr_img = hr_batch, lr_batch
            hr_img, lr_img = hr_img.cuda(), lr_img.cuda()
            sr_img = G(lr_img)

            # PSNR
            mse_metrics = torch.mean((sr_img * 1.0 - hr_img * 1.0) ** 2 , dim=[1, 2, 3])
            batch_psnr = 10 * torch.log10_(1.0 ** 2 / mse_metrics)
            test_psnr_result = torch.cat((test_psnr_result, batch_psnr))

            # SSIM
            batch_ssim = ssim(sr_img,hr_img,data_range=1.0, reduction='none')
            test_ssim_result = torch.cat((test_ssim_result, batch_ssim))

		    # Free GPU
            del hr_img, lr_img, sr_img,mse_metrics,batch_psnr,batch_ssim,
            torch.cuda.empty_cache()

        return torch.mean(test_psnr_result),torch.mean(test_ssim_result)
     

def evaluate(D,G,criterion_D,criterion_G,hr_loader,lr_loader):
    G.eval() # From train mode to eval mode
    D.eval()
    with torch.no_grad():
        val_psnr_result = torch.tensor([]).cuda()
        val_ssim_result = torch.tensor([]).cuda()
        val_epoch_loss_g = []
        val_epoch_loss_d = []
        for (batch, hr_batch), lr_batch in tqdm(zip(enumerate(hr_loader), lr_loader),total=len(lr_loader)):
            #hr_img, lr_img = hr_batch.cuda(), lr_batch.cuda()
            hr_img, lr_img = hr_batch, lr_batch
            hr_img, lr_img = hr_img.cuda(), lr_img.cuda()
            sr_img = G(lr_img)
            # Loss
            real,fake = torch.full(size=(len(hr_img),), fill_value=1.0, dtype=torch.float, device=device),torch.full(size=(len(hr_img),), fill_value=0.0, dtype=torch.float, device=device)
            output_real = D(hr_img).view(-1)
            err_D_real = criterion_D(output_real, real)
            output_fake = D(sr_img).view(-1)
            err_D_fake = criterion_D(output_fake, fake)
            err_D = err_D_real+err_D_fake
      
            adversarial_loss, content_loss = criterion_G(sr_img, hr_img, output_fake,real)
            err_G = 1e-3 * adversarial_loss + content_loss

            val_epoch_loss_d.append(err_D.item())
            val_epoch_loss_g.append(err_G.item())

			# PSNR
            mse_metrics = torch.mean((sr_img * 1.0 - hr_img * 1.0) ** 2 , dim=[1, 2, 3])
            batch_psnr = 10 * torch.log10_(1.0 ** 2 / mse_metrics)
            val_psnr_result = torch.cat((val_psnr_result, batch_psnr))

			# SSIM
            batch_ssim = ssim(sr_img,hr_img,data_range=1.0, reduction='none')
            val_ssim_result = torch.cat((val_ssim_result, batch_ssim))

		    # Free GPU
            del hr_img, lr_img, sr_img, real, fake, output_real, output_fake, err_D_real, err_D_fake, err_G, adversarial_loss, content_loss, err_D,mse_metrics,batch_psnr,batch_ssim,
            torch.cuda.empty_cache()

        val_psnr_result = val_psnr_result.to('cpu')
        val_ssim_result = val_ssim_result.to('cpu')
        loss_g,loss_d,v_psnr,v_ssim= np.mean(val_epoch_loss_g),np.mean(val_epoch_loss_d),torch.mean(val_psnr_result),torch.mean(val_ssim_result)

    return loss_g,loss_d,v_psnr,v_ssim
