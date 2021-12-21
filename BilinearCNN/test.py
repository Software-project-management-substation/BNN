import torch
import torch.nn as nn
# import torch.optim
# import torch.utils.data
import torchvision
import os
import bilinear_resnet
import CUB_200

from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from torchvision import transforms, utils,datasets
from PIL import Image
import pandas as pd
import numpy as np
#过滤警告信息
import warnings
warnings.filterwarnings("ignore")

# base_lr = 0.1
batch_size = 48
# num_epochs = 50
# weight_decay = 1e-8
num_classes = 10
resize = 448
cub200_path = 'data'
save_model_path = 'D:\\Code\\BilinearCNN\\data\\model_saved\\CIFAR_10'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = bilinear_resnet.BCNN(num_classes, pretrained=False).to(device)
model.load_state_dict(torch.load(os.path.join(save_model_path,
                                              'resnet34_CIFAR_10_fine_tuning_epoch_6_acc_99.4083.pth'),
                                 map_location=lambda storage, loc: storage))

test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(resize),
                                                     torchvision.transforms.CenterCrop(resize),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize([0.5231, 0.5332, 0.5011],
                                                                                      [0.2299, 0.2238, 0.2304])])
# torchvision.transforms.Normalize([0.4856, 0.4994, 0.4324],[0.1817, 0.1811, 0.1927]

test_data = datasets.ImageFolder('D://Code//BilinearCNN//data//val', transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
# 画图
loader_iter=iter(test_loader)
imags, labs=loader_iter.next()

# 开始性能测试
def test_accuracy(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)

            _, prediction = torch.max(outputs.data, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)
        model.train()
        return 100 * correct / total

# 展示输入数据
import matplotlib.pyplot as plt
plt.figure()
plt.title('input images')
for i in range(20):
    im=imags[i,:]
    imt=torch.transpose(torch.transpose(im,1,2),0,2)
    plt.subplot(4,5,i+1)
    plt.title(str(i))
    plt.imshow(imt)
plt.show()
    
test_acc = test_accuracy(model, test_loader)
print('the prediction accuracy is: '+str(test_acc))
