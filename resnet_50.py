import keras
import cv2
import os
import numpy as np
import glob
import tqdm
from random import shuffle
from keras.applications import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import *
from keras import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy


# control gpu memery
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))


class MyModel:
    def __init__(self, model_path='model.h5', root_path='', models=ResNet50, pre=False):
        self.root_path = root_path
        self.model_path = model_path
        if root_path:
            self.class_name = os.listdir(root_path)
        else:
            self.class_name = ['律师证', '身份证', '文档', '常口现实库信息', '营业执照']
        print("classes:{}".format(self.class_name))
        self.classes = len(self.class_name)
        self.model = self.model(models)
        if os.path.exists(self.model_path) and pre:
            self.model.load_weights(self.model_path)

    def read_img(self, path):
        return cv2.resize(cv2.imread(path), (256, 256)).astype(np.float32) / 255.

    def read_img_label(self, path):
        images = []
        labels = []
        all_name = glob.glob(os.path.join(path, '*/*.jpg'))
        shuffle(all_name)
        td = tqdm.tqdm(all_name)
        for name in td:
            p = name.split('/')
            td.set_description(p[-2])
            images.append(self.read_img(name))
            labels.append(self.class_name.index(p[-2]))
        images = np.array(images)
        labels = np.array(labels)
        return images, to_categorical(labels, self.classes)

    def split_train_test(self, x, y, p=0.8):
        l = int(len(y) * p)
        return x[:l], y[:l], x[l:], y[l:]

    def model(self, models):
        model = models(include_top=False, pooling='avg', classes=self.classes)
        x = model.output
        x = Dense(64, activation='relu')(x)
        x = Dense(self.classes, activation='softmax')(x)
        model = Model(model.input, x)
        return model

    def train(self, aug=None):
        x, y = self.read_img_label(self.root_path)
        x_train, y_train, x_text, y_test = self.split_train_test(x, y)
        if aug:
            x_train = aug(x_train)
        model = self.model
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(1e-4), metrics=['acc'], )
        model.fit(x_train, y_train, epochs=10,
                  validation_data=(x_text, y_test))
        model.save_weights(self.model_path)

    def predict(self, path):
        return self.model.predict(np.expand_dims(self.read_img(path), axis=0))


def train():
    clf = MyModel(root_path='/home/lz/接收/证件', models=ResNet50)
    clf.train(aug=None)


def pre():
    clf = MyModel(root_path='/home/lz/接收/证件', pre=True)
    print(clf.class_name[np.argmax(clf.predict('/home/lz/接收/证件/律师证/0008 (2).jpg'))])
    print(clf.class_name[np.argmax(clf.predict('/home/lz/接收/证件/身份证/0001.jpg'))])
    print(clf.class_name[np.argmax(clf.predict('/home/lz/接收/证件/律师证/20190816_003_10.jpg'))])


if __name__ == "__main__":
    train()
    pre()
