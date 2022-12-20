import torch
from torch import nn
from torch.autograd import Variable

x = torch.FloatTensor([1, 1, 1, 1]).view(1, 4, 1, 1)   # .view(1, -1, 1, 1)

x = Variable(x)

print("Shape of x:", x.shape)    # torch.Size([1, 4, 1, 1])   # Batch Size, Channel, Height, Width



# 1) groups = 1

conv = nn.Conv2d(in_channels=4,
                 out_channels=2,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=1,  # the number of groups
                 bias=False)



# shape of convolution weight: [out channels, in channels/groups, kernel size, kernel size]
print("Shape of convolution weight:", conv.weight.data.shape)  # torch.Size([2, 4, 1, 1])
print(conv.weight.data)



output = conv(x)
print("Shape of output:", output.shape)     # torch.Size([1, 2, 1, 1])


print(output)



# 2) groups = 2

conv = nn.Conv2d(in_channels=4,
                 out_channels=2,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=2,  # the number of groups
                 bias=False)



# shape of convolution weight: [out channels, in channels/groups, kernel size, kernel size]

print("Shape of convolution weight:", conv.weight.data.shape)  # torch.Size([2, 2, 1, 1])
print(conv.weight.data)


output = conv(x)

print("Shape of output:", output.shape)     # torch.Size([1, 2, 1, 1])

print(output)





# 3) group 2, in_channels = 4

x = torch.FloatTensor([1, 1, 1, 1, 1, 1]).view(1, 6, 1, 1)   # .view(1, -1, 1, 1)

x = Variable(x)

print("Shape of x:", x.shape)    # torch.Size([1, 6, 1, 1])   # Batch Size, Channel, Height, Width



conv = nn.Conv2d(in_channels=6,

                 out_channels=3,

                 kernel_size=1,

                 stride=1,

                 padding=0,

                 groups=3,  # the number of groups

                 bias=False)

# shape of convolution weight: [out channels, in channels/groups, kernel size, kernel size]

print("Shape of convolution weight:", conv.weight.data.shape)  # torch.Size([3, 2, 1, 1])

print(conv.weight.data)



output = conv(x)

print("Shape of output:", output.shape)     # torch.Size([1, 3, 1, 1])

print(output)