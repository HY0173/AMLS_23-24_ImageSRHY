from tqdm import tqdm
import os
import numpy as np
import torch
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from piq import ssim, psnr
from dataset import Div2kDataset
from SRGAN import Generator,Discriminator
from loss import GeneratorLoss
from evaluate import evaluate,test


# Check the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define Model_initialization Function
def xavier_init_weights(model):
	if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv2d):
		torch.nn.init.xavier_uniform_(model.weight)
            
def train(device, High_Train_ROOT, Low_Train_ROOT, High_Val_ROOT, Low_Val_ROOT, High_Test_ROOT, Low_Test_ROOT, BATCH_SIZE, lr, EPOCH, out_path):
    # Load Data: dataset --> dataloader
    print("Loading Training Data...")
    data_train_hr = Div2kDataset(data_dir=High_Train_ROOT,transform=Compose([CenterCrop(400),ToTensor()]))
    data_train_lr = Div2kDataset(data_dir=Low_Train_ROOT,transform=Compose([CenterCrop(100),ToTensor()]))
    hr_train_loader = DataLoader(dataset=data_train_hr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
    lr_train_loader = DataLoader(dataset=data_train_lr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)

    print("Loading Validation Data...")
    data_val_hr = Div2kDataset(data_dir=High_Val_ROOT)
    data_val_lr = Div2kDataset(data_dir=Low_Val_ROOT)
    hr_val_loader = DataLoader(dataset=data_val_hr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
    lr_val_loader = DataLoader(dataset=data_val_lr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
    
    print("Loading Testing Data...")
    data_test_hr = Div2kDataset(data_dir=High_Test_ROOT,transform=Compose([CenterCrop(400),ToTensor()]))
    data_test_lr = Div2kDataset(data_dir=Low_Test_ROOT,transform=Compose([CenterCrop(100),ToTensor()]))
    hr_test_loader = DataLoader(dataset=data_test_hr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)
    lr_test_loader = DataLoader(dataset=data_test_lr, shuffle=False, batch_size=BATCH_SIZE, drop_last=False)

    # Load Model and initialization
    print("Loading Model...")
    G = Generator().to(device)
    D = Discriminator().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=20*lr, betas=(0.5,0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

    G.apply(xavier_init_weights)
    D.apply(xavier_init_weights)
    summary(G, input_size=(3, 100, 100), batch_size=BATCH_SIZE, device=str(device))
    summary(D, input_size=(3, 400, 400), batch_size=BATCH_SIZE, device=str(device))

    # Define Loss Function
    criterion_G = GeneratorLoss().cuda()
    criterion_D = torch.nn.BCELoss().cuda()

    results = {'d_loss': [], 'g_loss': [], 'psnr': [], 'ssim': []}

    # Start Training
    print("Start Training...")
    for e in range(EPOCH):
        print(f"\nEpoch: {e+1}")
        train_epoch_loss_g = []
        train_epoch_loss_d = []
        psnr_result = torch.tensor([]).cuda()
        ssim_result = torch.tensor([]).cuda()

        for (batch, hr_batch), lr_batch in tqdm(zip(enumerate(hr_train_loader), lr_train_loader),total=len(lr_train_loader)):
            G.train()
            D.train()
            ''' (0) Data --> GPU '''
            hr_img, lr_img = hr_batch, lr_batch
            hr_img, lr_img = hr_img.cuda(), lr_img.cuda()

            '''(1) Update Discriminator
                    --> Maximize log(D(real)) + log(1-D(G(fake)))
            '''
            optimizer_D.zero_grad()
            # we want the output of real_img is 1.0
            real = torch.full(size=(len(hr_img),), fill_value=1.0, dtype=torch.float, device=device)
            output_real = D(hr_img).view(-1)
            err_D_real = criterion_D(output_real, real)
            err_D_real.backward()

            # we want the output of fake_img is 0.0
            fake = torch.full(size=(len(hr_img),), fill_value=0.0, dtype=torch.float, device=device)
            sr_img = G(lr_img)      # Generate fake images with low_img
            output_fake = D(sr_img.detach()).view(-1)
            err_D_fake = criterion_D(output_fake, fake)
            err_D_fake.backward()
            err_D = err_D_real+err_D_fake
            if err_D.item()>0.1:
                optimizer_D.step()

            '''
            (2) Update Generator
                --> Minimize 1e-3 * Adversarial Loss + Content Loss
            '''
            optimizer_G.zero_grad()

            output_fake = D(sr_img).view(-1)
            adversarial_loss, content_loss = criterion_G(sr_img, hr_img, output_fake,real)
            err_G = 1e-3 * adversarial_loss + content_loss
            err_G.backward()
            optimizer_G.step()

            # Record train_loss, psnr, ssim of each batch
            train_epoch_loss_d.append(err_D.item())
            train_epoch_loss_g.append(err_G.item())
            mse_metrics = torch.mean((sr_img * 1.0 - hr_img * 1.0) ** 2 , dim=[1, 2, 3])
            batch_psnr = 10 * torch.log10_(1.0 ** 2 / mse_metrics)
            #batch_psnr = psnr(sr_img,hr_img,data_range=1.0, reduction='none')
            batch_ssim = ssim(sr_img,hr_img,data_range=1.0, reduction='none')
            psnr_result = torch.cat((psnr_result, batch_psnr))
            ssim_result = torch.cat((ssim_result, batch_ssim))

            # Free up GPU memory
            del hr_img, lr_img, sr_img, real, fake, output_real, output_fake, err_D_real, err_D_fake, err_G,adversarial_loss, content_loss, err_D, mse_metrics,batch_psnr,batch_ssim,
            torch.cuda.empty_cache()

        # Record avgloss, avgpsnr of each epoch
        train_epoch_avg_loss_g = np.mean(train_epoch_loss_g)
        train_epoch_avg_loss_d = np.mean(train_epoch_loss_d)
        train_epoch_avg_psnr = torch.mean(psnr_result)
        train_epoch_avg_ssim = torch.mean(ssim_result)
        
        results['g_loss'].append(train_epoch_avg_loss_g)
        results['d_loss'].append(train_epoch_avg_loss_d)
        results['psnr'].append(train_epoch_avg_psnr)
        results['ssim'].append(train_epoch_avg_ssim)
        
        print(f'Epoch {e + 1}, Generator Train Loss: {train_epoch_avg_loss_g:.4f}, '
            f'Discriminator Train Loss: {train_epoch_avg_loss_d:.4f}, PSNR: {train_epoch_avg_psnr:.4f}, SSIM: {train_epoch_avg_ssim:.4f}')

        # Eval
        print('Evaluate on Validation set...')
        val_gloss,val_dloss,val_psnr,val_ssim = evaluate(D,G,criterion_D,criterion_G,hr_val_loader,lr_val_loader)
        print(f'Generator Val Loss: {val_gloss:.4f}, Discriminator Val Loss: {val_dloss:.4f}, Val_PSNR: {val_psnr:.4f}, Val_SSIM: {val_ssim:.4f}')

        del train_epoch_avg_psnr, train_epoch_avg_ssim,
        torch.cuda.empty_cache()
        
        # Save Checkpoints
        if (e+1)%5==0:
            out_path_G = out_path+"/Epoch_"+str(e+1)+'_G.pth'
            out_path_D = out_path+"/Epoch_"+str(e+1)+'_D.pth'
            torch.save(G.state_dict(),out_path_G)
            torch.save(D.state_dict(),out_path_D)
    
    print("Completed!")
    test_psnr,test_ssim = test(D,G,hr_test_loader,lr_test_loader)
    print(f'Test_PSNR: {test_psnr:.4f}, Test_SSIM: {test_ssim:.4f}')
    