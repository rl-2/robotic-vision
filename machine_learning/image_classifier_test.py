#==========================================================================  
# A Basic Image Classifier -- testing part
# 
# The code was adapted from PyTorch.org
# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# 
# The code initially served for UCSB ART185 Intelligent Machine Vision
#
# Authors: Jieliang (Rodger) Luo
#
# April 22nd, 2018
#==========================================================================

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

# labels for the images 
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# build the nerual network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3-channel images, 6 outputs, 5x5 kernel 
        self.pool = nn.MaxPool2d(2, 2) # pooling layer is to reduce the size of the representations
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # fc stands for "fully connected"
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # ReLU (Rectified Linear Units) increases the nonlinear properties
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
model = torch.load("classifier_model.pkl")

img = Image.open('image.jpg')

crop_size = 0
if img.width < img.height:
	crop_size = img.width
else:
	crop_size = img.height

# image processing to make it readable for torch
preprocess = transforms.Compose([
	transforms.CenterCrop(crop_size),
	transforms.Resize(32),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

img_tensor = preprocess(img).unsqueeze(0) # add batch dim 
outputs = model(Variable(img_tensor))

print(outputs)

_, predicted_index = torch.max(outputs.data, 1)

print("The image is {}".format(classes[predicted_index[0]]))

