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
    init_centers = input[init_k(length, k)]
    input_device = cuda.to_device(input)  # to gpu
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

def init_k(length, k):
    inin_nums = np.random.randint(0, length-1, k)
    return inin_nums



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
    imageio.imwrite("./output.png", output)
    end = time.time()
    print("total time(s): {}".format(end-start))