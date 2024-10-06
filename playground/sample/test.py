import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import Function

import matplotlib.pyplot as plt

import random
from skimage.transform import resize


batch_size = 128
learning_rate = 0.0001
num_epoch = 10


def one_sided_padding(x):
    rand1 = random.randrange(0,15,3)
    rand2 = random.randrange(0,15,3)

    zero = np.zeros(shape=[28,28,1])
    zero[rand1:rand1+12,rand2:rand2+12,:]=np.asarray(x).reshape(12,12,1)
    return zero


mnist_train = dset.MNIST("./", train=True,
                         transform=transforms.Compose([
                            transforms.RandomCrop(22),
                            transforms.Resize(12),
                            transforms.Lambda(one_sided_padding),
                            transforms.ToTensor(),
                         ]),
                         target_transform=None,
                         download=True)

mnist_test = dset.MNIST("./", train=False,
                        transform=transforms.Compose([
                            transforms.RandomCrop(22),
                            transforms.Resize(12),
                            transforms.Lambda(one_sided_padding),
                            transforms.ToTensor(),
                        ]),
                        target_transform=None,
                        download=True)

train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)


class CNN(nn.Module):
    def __init__(self, num_feature=32):
        super(CNN, self).__init__()
        self.num_feature = num_feature

        self.layer = nn.Sequential(
            nn.Conv2d(1, self.num_feature, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature),
            nn.ReLU(),
            nn.Conv2d(self.num_feature, self.num_feature * 2, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature * 2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(self.num_feature * 2, self.num_feature * 4, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature * 4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(self.num_feature * 4, self.num_feature * 8, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature * 8),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_feature * 8 * 7 * 7, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):

                # Kaming Initialization
                init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.Linear):

                # Kaming Initialization
                init.kaiming_normal(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        out = self.layer(x)
        out = out.view(x.size()[0], -1)
        out = self.fc_layer(out)

        return out


model = nn.DataParallel(CNN().cuda())

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(num_epoch):
    model.train()
    for j, [image, label] in enumerate(train_loader):
        x = Variable(image).cuda()
        y_ = Variable(label).cuda()

        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

    top_1_count = torch.FloatTensor([0])
    total = torch.FloatTensor([0])
    model.eval()
    for image, label in test_loader:
        x = Variable(image, volatile=True).cuda()
        y_ = Variable(label).cuda()

        output = model.forward(x)

        values, idx = output.max(dim=1)
        top_1_count += torch.sum(y_ == idx).float().cpu().data

        total += label.size(0)

    print("Test Data Accuracy: {}%".format(100 * (top_1_count / total).numpy()))
    if (top_1_count / total).numpy() > 0.98:
        break


class GuidedBackpropRelu(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[grad_input < 0] = 0
        grad_input[input < 0] = 0
        return grad_input


guided_relu = GuidedBackpropRelu.apply


class GuidedReluModel(nn.Module):
    def __init__(self, model, to_be_replaced, replace_to):
        super(GuidedReluModel, self).__init__()
        self.model = model
        self.to_be_replaced = to_be_replaced
        self.replace_to = replace_to
        self.layers = []
        self.output = []

        for m in self.model.modules():
            if isinstance(m, self.to_be_replaced):
                self.layers.append(self.replace_to)
                # self.layers.append(m)
            elif isinstance(m, nn.Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                self.layers.append(m)
            elif isinstance(m, nn.Linear):
                self.layers.append(m)
            elif isinstance(m, nn.AvgPool2d):
                self.layers.append(m)

        for i in self.layers:
            print(i)

    def reset_output(self):
        self.output = []

    def hook(self, grad):
        out = grad[:, 0, :, :].cpu().data  # .numpy()
        print("out_size:", out.size())
        self.output.append(out)

    def get_visual(self, idx, original_img):
        grad = self.output[0][idx]
        return grad

    def forward(self, x):
        out = x
        out.register_hook(self.hook)
        for i in self.layers[:-3]:
            out = i(out)
        out = out.view(out.size()[0], -1)
        for j in self.layers[-3:]:
            out = j(out)
        return out
