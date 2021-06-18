from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np


def load_train():
    train_data = []
    train_data.append(img_to_array(load_img('../images/A.png')))
    for i in range(19):
        train_data.append(img_to_array(load_img(f'../images/A{i + 1}.png')))
    train_data.append(img_to_array(load_img('../images/B.png')))
    for i in range(19):
        train_data.append(img_to_array(load_img(f'../images/B{i + 1}.png')))
    train_data.append(img_to_array(load_img('../images/C.png')))
    for i in range(19):
        train_data.append(img_to_array(load_img(f'../images/c{i + 1}.png')))
    train_data.append(img_to_array(load_img('../images/d.png')))
    for i in range(19):
        train_data.append(img_to_array(load_img(f'../images/d{i + 1}.png')))
    train_data = np.array(train_data)
    return train_data


def load_label():
    train_labels = []
    for i in ['A', 'B', 'C', 'D']:
        for j in range(20):
            train_labels.append(i)
    train_labels = np.array(train_labels)
    return train_labels


def label_to_numpy(labels):
    final_labels = np.zeros((len(labels), 4))
    for i in range(len(labels)):
        label = labels[i]
        if label == 'A':
            final_labels[i, :] = np.array([1, 0, 0, 0])
        if label == 'B':
            final_labels[i, :] = np.array([0, 1, 0, 0])
        if label == 'C':
            final_labels[i, :] = np.array([0, 0, 1, 0])
        if label == 'D':
            final_labels[i, :] = np.array([0, 0, 0, 1])
    return final_labels


def load_test_data():
    test_data = []
    for i in ['a', 'b', 'c', 'd']:
        for j in range(5):
            test_data.append(img_to_array(load_img(f'../images/t{i}{j}.png')))
    test_data = np.array(test_data)
    return test_data


def load_test_labels():
    test_labels = []
    for i in ['a', 'b', 'c', 'd']:
        for j in range(5):
            test_labels.append(i)
    test_labels = np.array(test_labels)
    return test_labels
