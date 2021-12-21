import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import os
import bilinear_resnet
import CUB_200

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from torchvision import transforms, utils,datasets
from PIL import Image
import pandas as pd
import numpy as np
#过滤警告信息
import warnings
warnings.filterwarnings("ignore")

base_lr = 0.001
batch_size = 16
num_epochs = 50
weight_decay = 1e-5
num_classes = 10
#resize = 448
resize = 448
cub200_path = 'data'
save_model_path = 'data\model_saved\CIFAR_10'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    model = bilinear_resnet.BCNN(num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(save_model_path,
                                                  'resnet34_CIFAR_10_train_fc_epoch_16_acc_97.0414.pth'),
                                                  map_location=lambda storage, loc: storage))
    model_d = model.state_dict()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    train_transform = transforms.Compose([transforms.Resize(resize),
                                          transforms.CenterCrop(resize),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5231, 0.5332, 0.5011],
                                                               [0.2299, 0.2238, 0.2304])])
    test_transform = transforms.Compose([torchvision.transforms.Resize(resize),
                                         transforms.CenterCrop(resize),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5231, 0.5332, 0.5011],
                                                              [0.2299, 0.2238, 0.2304])])

    # train_data = CUB_200.CUB_200(cub200_path, train=True, transform=train_transform)
    # test_data = CUB_200.CUB_200(cub200_path, train=False, transform=test_transform)
    train_data = datasets.ImageFolder('D://Code//BilinearCNN//data//train', transform=train_transform)
    test_data = datasets.ImageFolder('D://Code//BilinearCNN//data//test', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    print('Start fine-tuning...')
    best_acc = 0.
    best_epoch = 0
    end_patient = 0
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        epoch_loss = 0.
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            _, prediction = torch.max(outputs.data, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

            print('Epoch %d: Iter %d, Loss %g' % (epoch + 1, i + 1, loss))
        train_acc = 100 * correct / total
        print('Testing on test dataset...')
        test_acc = test_accuracy(model, test_loader)
        print('Epoch [{}/{}] Loss: {:.4f} Train_Acc: {:.4f}  Test_Acc: {:.4f}'
              .format(epoch + 1, num_epochs, epoch_loss, train_acc, test_acc))
        scheduler.step(test_acc)
        if test_acc > best_acc:
            model_file = os.path.join(save_model_path, 'resnet34_CIFAR_10_fine_tuning_epoch_%d_acc_%g.pth' %
                                      (best_epoch, best_acc))
            if os.path.isfile(model_file):
                os.remove(os.path.join(save_model_path, 'resnet34_CIFAR_10_fine_tuning_epoch_%d_acc_%g.pth' %
                                       (best_epoch, best_acc)))
            end_patient = 0
            best_acc = test_acc
            best_epoch = epoch + 1
            print('The accuracy is improved, save model')
            torch.save(model.state_dict(), os.path.join(save_model_path,
                                                        'resnet34_CIFAR_10_fine_tuning_epoch_%d_acc_%g.pth' %
                                                        (best_epoch, best_acc)))
        else:
            end_patient += 1

        if end_patient >= 10:
            break
    print('After the training, the end of the epoch %d, the accuracy %g is the highest' % (best_epoch, best_acc))


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


def main():

    train()

if __name__ == '__main__':
    main()
