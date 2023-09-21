import matplotlib.pyplot as plt
import torch
import torchvision
from from_dataset import load_dataset
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# 可视化函数
def draw(features, labels):
    pca = PCA(n_components=2)
    fig = plt.Figure(figsize=(10, 10))
    features_pca = pca.fit_transform(features)
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Dog vs Cat Classification (PCA)")
    plt.colorbar()
    plt.savefig("./scatter.png")

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = load_dataset("./dataset", is_npy=True)
    """使用ResNet提取图片特征"""
    train_data = train_data.transpose((0, 3, 1, 2))
    test_data = test_data.transpose((0, 3, 1, 2))

    model = torchvision.models.resnet18(pretrained=True)
    train_data = torch.tensor(train_data, dtype=torch.float32)
    if torch.cuda.is_available() == True:
        model = model.cuda()
        train_data = train_data.cuda()
    train_features = model(train_data)
    if torch.cuda.is_available() == True:
        train_features = train_features.cpu().detach().numpy()

    draw(train_features, train_labels)

    """构建SVM"""
    clf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
    """train"""
    clf.fit(train_features, train_labels)

    """predict"""
    test_data = torch.tensor(test_data, dtype=torch.float32)
    if torch.cuda.is_available() == True:
        test_data = test_data.cuda()
    test_features = model(test_data)
    if torch.cuda.is_available() == True:
        test_features = test_features.cpu().detach().numpy()
    predict = clf.predict(test_features)

    # 计算准确率
    accuracy = accuracy_score(test_labels, predict)
    print("Accuracy:", accuracy)

    # precision_score
    precision_score = precision_score(test_labels, predict)
    print("Precision_score:", precision_score)

    # recall score
    recall_score = recall_score(test_labels, predict)
    print("Recall score:", recall_score)


