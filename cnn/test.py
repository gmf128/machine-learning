import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import Model

"""定义一些超参数"""
batch_size = 64
epoch_size = 20


test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print("predicting ... ")
# 最大的正确率及对应的epoch值
max_correctness = 0
max_epoch = 0
for epoch in range(0, epoch_size):

    # 加载模型
    model = Model.myCNN()
    model.load_state_dict(torch.load('./models/checkpoint_{}.pth'.format(epoch)))
    model.eval()

    # test
    correct = 0
    total = 0

    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    correctness = 100 * correct / total
    print('测试集在epoch:{}准确率: {:.4f} %'.format(epoch, correctness))
    if correctness > max_correctness:
        max_correctness = correctness
        max_epoch = epoch

print("最大正确率:{:.4f}".format(max_correctness))