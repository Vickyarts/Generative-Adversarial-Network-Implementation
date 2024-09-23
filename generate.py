import torch
import cv2
import random
from model import *


G = Generator()
G.load_state_dict(torch.load("weights/car_weights.pth")["Generator"])
noise = torch.randn((64, 100, 1, 1))
images = G(noise).detach()


for i, image in enumerate(images):
    image = image.permute((1, 2, 0))
    image = image.numpy()
    cv2.imwrite(f"results/generated-{i+1}.jpg", image * 255)
