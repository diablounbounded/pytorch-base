import torch
import torchvision.datasets
from torch import nn
import torch.nn.functional as F
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

train_dataset = torchvision.datasets.CIFAR10(root="../data", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)


class module(nn.Module):
    def __init__(self):
        super(module, self).__init__()
        """self.conv1=nn.Conv2d(1,20,5)
        self.conv2 = nn.Conv2d(20, 20, 5)

"""
        self.mode = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        output = self.mode(x)
        return output


tu = module()
if torch.cuda.is_available():
    tu=tu.cuda()
"""writer=SummaryWriter("../logs")"""
step = 0
"""
for data in dataloader:
    imgs,targets=data
    """
"""writer.add_images("inputs",imgs,step)"""
"""
    output=tu(imgs)
    """
"""writer.add_images("outputs", output, step)""""""
    step=step+1
"""

print("训练测试集长度为：{}".format(len(train_dataset)))
print("hahah")

# 损失函数
loss_nn = nn.CrossEntropyLoss()
loss_nn=loss_nn.cuda()
learning_rate = 0.001
optimizer = torch.optim.SGD(tu.parameters(), lr=learning_rate)
# 记录训练的次数
totol_train_step = 0
# 记录测试的次数
totol_test_step = 0
writer = SummaryWriter("gpu_logstest4")
epoch = 10
for i in range(epoch):
    print("第{}训练开始了".format(i + 1))
    # 训练框架
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs=imgs.cuda()
            targets=targets.cuda()
        output = tu(imgs)
        loss1 = loss_nn(output, targets)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        totol_train_step = totol_train_step + 1
        if (totol_train_step) % 100 == 0:
            print("训练次数：{}，损失；{}".format(totol_train_step, loss1.item()))
            writer.add_scalar("gpu_train_loss", loss1.item(), totol_train_step)

    # 测试框架
    totol_test_loss = 0
    totol_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output2 = tu(imgs)
            loss2 = loss_nn(output2, targets)
            totol_test_loss = totol_test_loss + loss2.item()
            accuracy=(output2.argmax(1)==targets).sum()
            totol_accuracy=totol_accuracy+accuracy
    print("测试集上的loss：{}".format(totol_test_loss))
    print("测试集上的正确率：{}".format(totol_accuracy/len(test_dataset)))
    writer.add_scalar("gpu_test_loss", loss2.item(), totol_test_step)
    writer.add_scalar("test_accuracy",totol_accuracy/len(test_dataset),totol_test_step)
    totol_test_step = totol_test_step + 1

writer.close


