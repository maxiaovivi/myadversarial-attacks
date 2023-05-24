import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
import torchattacks
import robustbench
from robustbench.utils import load_model
from utils import imshow, get_pred
from torchattacks import PGD
device = "cuda"
model = load_model('Standard', norm='Linf').to(device)



atk1 = torchattacks.FGSM(model, eps=8/255)
atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, random_start=True)
atk = torchattacks.MultiAttack([atk1, atk2])
print(atk)

# Define the transformations: Convert the image to PyTorch Tensor and normalize with mean and standard deviation of CIFAR-10 dataset.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load images, perform adversarial attack, and save the adversarial examples
for i in range(1, 1001):
    img_path = f"/home/mxvivi/DATA/nvme/attacker/dataset/images/{i}.png"
    img = Image.open(img_path).convert("RGB")  # Ensure image is in RGB format
    img_tensor = transform(img).unsqueeze(0).to(device)  # Convert image to tensor and add batch dimension

    # Get prediction of the model as target
    labels = model(img_tensor).argmax(dim=1)
    
    # Repeat the labels to match the batch size of img_tensor
    labels = labels.repeat(img_tensor.size(0))

    # Clamp the images tensor to [0, 1] before feeding it to the attack
    img_tensor = torch.clamp(img_tensor, min=0, max=1)

    # Perform adversarial attack
    adv_img_tensor = atk(img_tensor, labels)

    # Convert back to PIL image and save
    adv_img_tensor = adv_img_tensor.squeeze().cpu().detach()  # Remove batch dimension and move to CPU
    adv_img = transforms.ToPILImage()(adv_img_tensor)
    adv_img.save(f"/home/mxvivi/DATA/nvme/attacker/adv/{i}.png")
