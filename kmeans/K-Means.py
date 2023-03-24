import time
import imageio
import numpy as np

# 图片预处理
def pre_process(image):
    # 1. squueze the image to [-1, 3]
    input = np.reshape(image, [-1, 3])
    input = input/255
    return input

# k-means算法
def kMeans(input, k):
    '''

    :param input: [width * heught, 3] tensor
    :param k: num of clusters
    :return:  [width * heught, 3] tensor
    '''
    epsilon = 1e-4  # 误差量用于判断循环终止
    length = input.shape[0]
    init_centers = input[init_k(length, k)]    # 初始的随机中心

    new_centers = init_centers  # 更新的中心
    count = 0
    belongs = np.zeros(length).astype(int)  # 存储每个像素的所属类
    while(1):
        loop_begin = time.time()    # 用于衡量运行时间，以便与后续gpu实现做对比
        tmp = update(input, new_centers, k, length, belongs)
        loss = np.power(tmp-new_centers, 2).sum()  # 计算误差
        if loss < epsilon:
            new_centers = tmp
            break
        new_centers = tmp # update
        loop_end = time.time()
        print("loop: {} done | loss: {} | using time(s): {}".format(count, loss, (loop_end- loop_begin)))
        count += 1
    for i in range(0, length):
        input[i] = new_centers[belongs[i]]
    output = np.floor(input*255.0).astype('uint8')
    return output

def init_k(length, k):
    '''
      @param: length: 数组长度
      @param: k:聚类数
      @return: 长度为k的数组
      '''
    inin_nums = np.random.randint(0, length-1, k)
    return inin_nums

# 更新聚类中心算法
def update(input, centers, k, length, belongs):
    count = np.ones(k)
    colors = np.zeros([k, 3])+centers
    """计算最小距离"""
    for i in range(0, length):
        min_dist = 100
        min_cnum = 0
        for j in range(0, k):
            if dist(input[i], centers[j]) < min_dist:
                min_dist = dist(input[i], centers[j])
                min_cnum = j
        count[min_cnum] += 1
        """更新所属聚类"""
        belongs[i] = min_cnum
        for w in range(0, 3):
            colors[min_cnum][w] += input[i][w]
    """更新聚类中心"""
    for r in range(0, k):
        colors[r] = colors[r] / count[r]
    return colors

def dist(i, j):
    '''
      @param i,j: RGB vector
      '''
    return np.power(i-j, 2).sum()/3

if __name__ == '__main__':
    start = time.time()
    # 读入图片
    image = imageio.imread("./input.png")
    width = image.shape[0]
    height = image.shape[1]
    # 预处理
    input = pre_process(image)
    # 聚类算法
    output = kMeans(input, 5)
    output = np.reshape(output, [width, height, 3])
    # 输出图片
    imageio.imwrite("./output.png", output)
    end = time.time()
    print("total time(s): {}".format(end-start))