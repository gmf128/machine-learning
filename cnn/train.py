import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import Model
import tqdm

"""首先定义一些超参数"""
learning_rate = 0.01
batch_size = 64
epoch_size = 20

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

# 加载模型定义
model = Model.myCNN()
model.cuda()
# 定义loss与优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().cuda()

# 训练模型
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#train

loop = tqdm.tqdm(range(0, epoch_size))

for epoch in loop:
    for i, (images, labels) in enumerate(train_loader):
        # use gpu
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        output = model(images)
        loss = criterion(output, labels)
        # output loss
        loop.set_postfix(loss=loss)
        # 清零grad
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
    outdir = format('./models/checkpoint_{}.pth'.format(epoch))
    torch.save(model.state_dict(), outdir)