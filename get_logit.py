import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, vgg16
from PIL import Image
import torch.nn.functional as F
import json
import urllib

# Download the imagenet category list
categories_link = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
with urllib.request.urlopen(categories_link) as url:
    categories = json.loads(url.read().decode())

# 定义transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the pretrained models
resnet = resnet50(pretrained=True).eval()
vgg = vgg16(pretrained=True).eval()

# 图片路径
image_folder = '/home/mxvivi/DATA/nvme/attacker/dataset/images/'

# 如果有GPU，则将模型转移到GPU上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet.to(device)
vgg.to(device)

# Create an ordered list of image file names
num_images = 1000  # replace with your actual number of images
image_files = [f"{i}.png" for i in range(1, num_images+1)]

# 进行预测
for i, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs_resnet = resnet(image)
        outputs_vgg = vgg(image)
        
    # 加权融合
    outputs = 0.5 * outputs_resnet + 0.5 * outputs_vgg
    
    _, preds = torch.max(outputs, 1)
    pred_class = categories[preds.item()]
    print(f'Image {i+1} is predicted as {pred_class} with ImageNet index: {preds.item()}.')
