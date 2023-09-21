import time
import imageio
import numpy as np
from numba import cuda

# 图片预处理
def pre_process(image):
    # 1. squeeze the image to [-1, 3]
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

    epsilon = 1e-4  # used to judge when to break the loop
    length = input.shape[0]  # num of pixels
    input_device = cuda.to_device(input)  # to gpu
    init_centers = input[init_kpp(length, k, input_device)]
    new_centers = init_centers
    counting = 0
    belongs = np.zeros(length).astype(int)  # denote the cluster # of each pixel
    belongs = cuda.to_device(belongs)

    while(1):
        loop_begin = time.time()

        # classify the pixels using gpu

        # refreshing the relations matrix
        relations = np.zeros([k, length]).astype(int)
        # to gpu
        relations = cuda.to_device(relations)
        new_centers = cuda.to_device(new_centers)
        # calculating the distances and classify
        classify[1024, 1024](input_device, new_centers, k, length, belongs, relations)
        cuda.synchronize()
        # synchronize
        relations = relations.copy_to_host()
        new_centers = new_centers.copy_to_host()
        # update (using matrix-product)
        count = np.ones(k)
        colors = np.matmul(relations, input)
        for i in range(0, k):
            count[i] += relations[i].sum()
        for i in range(0, k):
            colors[i] = colors[i]/count[i]

        loss = np.power(colors-new_centers, 2).sum()
        if loss < epsilon:
            break
        new_centers = np.zeros([k, 3])+colors
        loop_end = time.time()
        print("loop: {} done | loss: {} | using time(s): {}".format(counting, loss, (loop_end- loop_begin)))
        counting += 1

    # after loop
    belongs = belongs.copy_to_host()
    for i in range(0, length):
        input[i] = new_centers[belongs[i]]
    output = np.floor(input*255.0).astype('uint8')
    return output

def RWS(possibilities, r):
    """轮盘算法"""
    q = 0  # 累计概率
    for i in range(1, possibilities.shape[0] + 1):
        q += possibilities[i - 1]  # P[i]表示第i个个体被选中的概率
        if r <= q:  # 产生的随机数在m~m+P[i]间则认为选中了i
            return i

def init_kpp(length, k, input_device):
    inits = np.zeros(k).astype(int)
    init_first = np.random.randint(0, length - 1)
    """首先初始化一个结果"""
    inits[0] = init_first
    """循环得到k-1个结果"""
    for i in range(1, k):
        """首先计算概率"""
        possibilities = np.zeros(length)
        possibilities = cuda.to_device(possibilities)
        """gpu函数代替循环"""
        cal_p[1024, 1024](input_device, inits, length, possibilities, i)
        cuda.synchronize()
        possibilities = possibilities.copy_to_host()
        sum = possibilities.sum()
        possibilities /= sum
        """ 排序 """
        possibilities = np.sort(possibilities)
        """通过轮盘算法选出新的中心"""
        r = np.random.randint(0, 1)
        new = RWS(possibilities, r)
        inits[i] = k

    return inits

@cuda.jit()
def cal_p(input_device, inits, length, possibilities, has_inited):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < length:
        min_dist = 100
        min_num = 0
        for j in range(0, has_inited):
            if dist(input_device[i], input_device[inits[j]]) < min_dist:
                min_dist = dist(input_device[i], input_device[inits[j]])
        possibilities[i] = min_dist

@cuda.jit()
def classify(input, centers, k, length, belongs, relations):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < length:
        min_dist = 100
        min_cnum = 0
        for j in range(0, k):
            if dist(input[i], centers[j]) < min_dist:
                min_dist = dist(input[i], centers[j])
                min_cnum = j
        relations[min_cnum][i] = 1
        belongs[i] = min_cnum


@cuda.jit(device=True)
def dist(i, j):
    result = 0
    for w in range(0, 3):
        result += pow(i[w]-j[w], 2)
    return result


if __name__ == '__main__':
    start = time.time()
    image = imageio.imread("./input.png")
    width = image.shape[0]
    height = image.shape[1]
    input = pre_process(image)
    output = kMeans(input, 3)
    output = np.reshape(output, [width, height, 3])
    imageio.imwrite("./kpp_output.png", output)
    end = time.time()
    print("total time(s): {}".format(end-start))