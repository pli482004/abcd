from django.http import JsonResponse
from django.shortcuts import render
import json
from PIL import Image
from io import BytesIO

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np


def load_train():
    train_data = []
    train_data.append(img_to_array(load_img('images/A.png')))
    for i in range(19):
        train_data.append(img_to_array(load_img(f'images/A{i + 1}.png')))
    train_data.append(img_to_array(load_img('images/B.png')))
    for i in range(19):
        train_data.append(img_to_array(load_img(f'images/B{i + 1}.png')))
    train_data.append(img_to_array(load_img('images/C.png')))
    for i in range(19):
        train_data.append(img_to_array(load_img(f'images/c{i + 1}.png')))
    train_data.append(img_to_array(load_img('images/d.png')))
    for i in range(19):
        train_data.append(img_to_array(load_img(f'images/d{i + 1}.png')))
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
            test_data.append(img_to_array(load_img(f'images/t{i}{j}.png')))
    test_data = np.array(test_data)
    return test_data


def load_test_labels():
    test_labels = []
    for i in ['a', 'b', 'c', 'd']:
        for j in range(5):
            test_labels.append(i)
    test_labels = np.array(test_labels)
    return test_labels


# 64 by 64 pixels rgb
image_shape = (64, 64, 3)

# neural net parameters
nn_params = {'input_shape': image_shape, 'output_neurons': 4, 'loss': 'categorical_crossentropy',
             'output_activation': 'softmax'}

# monitor = ModelCheckpoint('./model.h5', monitor='val_accuracy', verbose=0, save_best_only=True,
# save_weights_only=False, mode='auto', save_freq='epoch')
monitor = ModelCheckpoint('./model.h5', monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False,
                          mode='auto')

# load images and labels from help.py
train_data = load_train()
train_labels = load_label()
test_data = load_test_data()
test_labels = load_test_labels()

# convert labels into numpy vectors (one-hot encoding)
train_labels = label_to_numpy(train_labels)
test_labels = label_to_numpy(test_labels)

# load the vgg network that is an 'expert' at 'imagenet' but do not include the FC layers
vgg_expert = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# we add the first 12 layers of vgg to our own model vgg_model
vgg_model = Sequential()
vgg_model.add(vgg_expert)

# and then add our own layers on top of it
vgg_model.add(GlobalAveragePooling2D())
vgg_model.add(Dense(1024, activation='relu'))
vgg_model.add(Dropout(0.3))
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dropout(0.3))
vgg_model.add(Dense(4, activation='softmax'))

# finally, we build the vgg model and turn it on so we can use it!
vgg_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.95),
                  metrics=['accuracy'])

vgg_model.fit(train_data, train_labels, epochs=15, shuffle=True)

# Create your views here.
from django.views.decorators.csrf import csrf_exempt


def index(request):
    return render(request, "letters/index.html", {
        "message": "Draw A, B, C, or D!"
    })


def test(request):
    return render(request, "letters/test.html", {
        "message": "Draw A, B, C, or D!"
    })


@csrf_exempt
def send(request):
    global response
    if request.method == "POST":
        ImageData = json.loads(request.body.decode())['data']
        import re
        import base64

        dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')
        ImageData = dataUrlPattern.match(ImageData).group(2)

        # If none or len 0, means illegal image data
        if ImageData == None or len(ImageData) == 0:
            print("failed image")
            pass

        im = Image.open(BytesIO(base64.b64decode(ImageData)))

        im = im.resize((64, 64), Image.ANTIALIAS)
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im, mask=im.split()[3])
        test_image = np.array([img_to_array(background)])

        x = np.argmax(vgg_model.predict(test_image), axis=-1)

        response = ""
        for i in range(4):
            if i in x:
                if i == 0:
                    response = 'A'
                if i == 1:
                    response = 'B'
                if i == 2:
                    response = 'C'
                if i == 3:
                    response = 'D'
    return JsonResponse({"hello": response}, status=200)
