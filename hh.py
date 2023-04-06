import torch
print("kkk")
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
writer=SummaryWriter("logss")
for i in range(100):
    writer.add_scalar("y=x",i,i)

writer.close()
print("kdd")
