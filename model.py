import torch 
import torch.nn as nn




class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 2, 0, bias=False),   #4x4
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),   #8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=True),    #16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),    #32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),     #64x64
            nn.Tanh()
        )

    def forward(self, random_noise):
        output = self.network(random_noise)
        return output
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),               # 256x256
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),             # 128x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),            # 64x64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),            # 32x32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 2, 0, bias=False),           # 16x16
            nn.Sigmoid()
        )

    def forward(self, image):
        output = self.network(image)
        return output.view(-1)
    
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Conv":
        m.weight.data.normal_(0.0, 0.2)
    elif classname == "BatchNorm":
        m.weight.data.normal_(1.0, 0.2)
        m.bias.data.fill_(0)
        
        
def saveCheckpoint(model_G, model_D, optimizer_G, optimizer_D, epoch=0, filepath='car_weights.pth'): 
    point = {
        'Generator': model_G.state_dict(), 
        'Discriminator': model_D.state_dict(), 
        'OptimizerG': optimizer_G.state_dict(), 
        'OptimizerD': optimizer_D.state_dict(),
        'epoch': epoch 
    }

    torch.save(point, filepath)
    
