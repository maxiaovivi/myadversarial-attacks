import sys
import torch
import torch.nn as nn
sys.path.insert(0, '..')
import torchattacks

sys.path.insert(0, '..')
import robustbench
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy

images, labels = load_cifar10(n_examples=5)
print('[Data loaded]')

device = "cuda"
model = load_model('Standard', norm='Linf').to(device)
images = images.to(device)  # 将images转移到GPU
labels = labels.to(device)  # 将labels转移到GPU
acc = clean_accuracy(model, images, labels)
print('[Model loaded]')
print('Acc: %2.2f %%'%(acc*100))

from torchattacks import PGD
from utils import imshow, get_pred

atk1 = torchattacks.FGSM(model, eps=8/255)
atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, random_start=True)
atk = torchattacks.MultiAttack([atk1, atk2])
print(atk)

adv_images = atk(images, labels)

idx = 0
pre = get_pred(model, adv_images[idx:idx+1], device)
imshow(adv_images[idx:idx+1], title="True:%d, Pre:%d"%(labels[idx], pre))
