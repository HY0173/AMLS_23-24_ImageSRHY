import torch
from torch import nn
from torchvision.models.vgg import vgg16,vgg19

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18]).eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = feature_extractor

    def forward(self, img):
        return self.feature_extractor(img)


class GeneratorLoss(nn.Module):
    '''Total Loss = 1e-3 * Adversarial Loss + Content Loss'''
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.mse_loss = nn.MSELoss()
        self.con_loss = nn.L1Loss()

    def forward(self, sr_img, hr_img, output_fake,real):
        # Adversarial Loss
        adversarial_loss = self.mse_loss(output_fake,real)

        # Content Loss
        content_loss = self.con_loss(self.feature_extractor(sr_img),self.feature_extractor(hr_img))

        return adversarial_loss, content_loss
