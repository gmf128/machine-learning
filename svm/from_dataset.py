import numpy as np
import tqdm
from skimage.io import imread
from skimage.transform import resize
import os

def load_dataset(path, is_npy=False):
    if is_npy:
        dataset_path = os.path.join(path, "saved_data")
        train_data = np.load(os.path.join(dataset_path, "train_data.npy"))
        validation_data = np.load(os.path.join(dataset_path, "validation_data.npy"))
        train_label = np.load(os.path.join(dataset_path, "train_label.npy"))
        validation_label = np.load(os.path.join(dataset_path, "validation_label.npy"))
        return train_data, train_label, validation_data, validation_label

    dataset_path = os.path.join(path, "train")
    save_path = os.path.join(path, "saved_data")
    """load"""
    train_data = []
    labels_train = []   # cat:0; dog:1
    validation_data = []
    labels_validation = []
    # cat:12500 dog:12500
    """自己拆分数据集"""
    # 80%: for training 20% for validation

    print("loading training data ...")
    loop = tqdm.tqdm(range(0, 100))
    for i in range(0, 100):
        dog_img_name = "dog."+str(i)+".jpg"
        cat_img_name = "cat."+str(i)+".jpg"
        dog_img = imread(os.path.join(dataset_path, dog_img_name))
        cat_img = imread(os.path.join(dataset_path, cat_img_name))
        dog_img = resize(dog_img, (224, 224))
        cat_img = resize(cat_img, (224, 224))
        train_data.append(dog_img)
        train_data.append(cat_img)
        labels_train.append(1)
        labels_train.append(0)
        loop.update()
    train_data = np.array(train_data)
    labels_train = np.array(labels_train)

    print("saving data ...")
    np.save(os.path.join(save_path, "train_data.npy"), train_data)
    np.save(os.path.join(save_path, "train_label.npy"), labels_train)
    """index:2i:dog ; 2i+1:cat"""
    print("loading validation data ... ")
    loop2 = tqdm.tqdm(range(0, 25))
    for i in range(100, 125):
        dog_img_name = "dog." + str(i) + ".jpg"
        cat_img_name = "cat." + str(i) + ".jpg"
        dog_img = imread(os.path.join(dataset_path, dog_img_name))
        cat_img = imread(os.path.join(dataset_path, cat_img_name))
        dog_img = resize(dog_img, (224, 224))
        cat_img = resize(cat_img, (224, 224))
        validation_data.append(dog_img)
        validation_data.append(cat_img)
        labels_validation.append(1)
        labels_validation.append(0)
        loop2.update()
    validation_data = np.array(validation_data)
    labels_validation = np.array(labels_validation)
    print("saving data ...")
    np.save(os.path.join(save_path, "validation_data.npy"), validation_data)
    np.save(os.path.join(save_path, "validation_label.npy"), labels_validation)
    return train_data, labels_train, validation_data, labels_validation