import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import glob
import io
import math
import time
import csv
import cv2

import keras.backend as K
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Sequential, Input, Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Reshape, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from scipy.stats import entropy
import tensorflow_addons as tfa
import os
import glob


def build_Unet():
    input_layer=Input(shape=(256,256,1))

    #인코더 생성
    encoder1=Conv2D(128,4,padding="same",strides=2)(input_layer)
    encoder1=LeakyReLU(alpha=0.2)(encoder1)

    encoder2=Conv2D(256,4,padding="same",strides=2)(encoder1)
    encoder2=BatchNormalization()(encoder2)
    encoder2=LeakyReLU(0.2)(encoder2)

    encoder3 = Conv2D(512, 4, padding="same", strides=2)(encoder2)
    encoder3 = BatchNormalization()(encoder3)
    encoder3= LeakyReLU(0.2)(encoder3)

    encoder4 = Conv2D(512, 4, padding="same", strides=2)(encoder3)
    encoder4 = BatchNormalization()(encoder4)
    encoder4 = LeakyReLU(0.2)(encoder4)

    encoder5 = Conv2D(512, 4, padding="same", strides=2)(encoder4)
    encoder5 = BatchNormalization()(encoder5)
    encoder5 = LeakyReLU(0.2)(encoder5)

    encoder6 = Conv2D(512, 4, padding="same", strides=2)(encoder5)
    encoder6 = BatchNormalization()(encoder6)
    encoder6 = LeakyReLU(0.2)(encoder6)

    encoder7 = Conv2D(512, 4, padding="same", strides=2)(encoder6)
    encoder7 = BatchNormalization()(encoder7)
    encoder7 = LeakyReLU(0.2)(encoder7)

    encoder8 = Conv2D(512, 4, padding="same", strides=2)(encoder7)
    encoder8 = BatchNormalization()(encoder8)
    encoder8 = LeakyReLU(0.2)(encoder8)

    #디코더 생성-상향 표본 추출
    decoder1=UpSampling2D((2,2))(encoder8)
    decoder1=Conv2D(512,4,padding="same")(decoder1)
    decoder1=BatchNormalization()(decoder1)
    decoder1=Dropout(0.5)(decoder1)
    decoder1=np.concatenate([decoder1, encoder7])
    decoder1=Activation("relu")(decoder1)

    decoder2 = UpSampling2D((2, 2))(decoder1)
    decoder2 = Conv2D(1024, 4, padding="same")(decoder2)
    decoder2 = BatchNormalization()(decoder2)
    decoder2 = Dropout(0.5)(decoder2)
    decoder2 = np.concatenate([decoder2, encoder6])
    decoder2 = Activation("relu")(decoder2)

    decoder3 =  UpSampling2D((2, 2))(decoder2)
    decoder3 = Conv2D(1024, 4, padding="same")(decoder3)
    decoder3 = BatchNormalization()(decoder3)
    decoder3 = Dropout(0.5)(decoder3)
    decoder3 = np.concatenate([decoder3, encoder5])
    decoder3 = Activation("relu")(decoder3)

    decoder4 =  UpSampling2D((2, 2))(decoder3)
    decoder4 = Conv2D(1024, 4, padding="same")(decoder4)
    decoder4 = BatchNormalization()(decoder4)
    decoder4 = Dropout(0.5)(decoder4)
    decoder4 = np.concatenate([decoder4, encoder4])
    decoder4 = Activation("relu")(decoder4)

    decoder5 =  UpSampling2D((2, 2))(decoder4)
    decoder5 = Conv2D(1024, 4, padding="same")(decoder5)
    decoder5 = BatchNormalization()(decoder5)
    decoder5 = Dropout(0.5)(decoder5)
    decoder5 = np.concatenate([decoder5, encoder3])
    decoder5 = Activation("relu")(decoder5)

    decoder6 = UpSampling2D((2, 2))(decoder5)
    decoder6 = Conv2D(512, 4, padding="same")(decoder6)
    decoder6 = BatchNormalization()(decoder6)
    decoder6 = Dropout(0.5)(decoder6)
    decoder6 = np.concatenate([decoder6, encoder2])
    decoder6 = Activation("relu")(decoder6)

    decoder7 = UpSampling2D((2, 2))(decoder6)
    decoder7 = Conv2D(256, 4, padding="same")(decoder7)
    decoder7 = BatchNormalization()(decoder7)
    decoder7 = Dropout(0.5)(decoder7)
    decoder7 = np.concatenate([decoder7, encoder1])
    decoder7 = Activation("relu")(decoder7)

    decoder8 = UpSampling2D((2, 2))(decoder7)
    decoder8= Conv2D(1, 4, padding="same")(decoder8)
    decoder8 = Activation("tanh")(decoder8)

    model=Model(inputs=[input_layer],outputs=[decoder8])
    return model


def build_discriminator():
    dis_model = Sequential()
    # [-1, 256,256,8]
    dis_model.add(
        Conv2D(8, (3, 3),strides=1,
               padding='same',
               input_shape=(256, 256, 1))
    )
    dis_model.add(LeakyReLU(alpha=0.2))

    # [-1, 128,128,16]
    dis_model.add(Conv2D(16, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3[-1,64,64,32]
    dis_model.add(Conv2D(32,  (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    # 4 [-1, 32,32,64]
    dis_model.add(Conv2D(64, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    # 4 [-1, 16,16,128]
    dis_model.add(Conv2D(128, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    # 4 [-1, 8,8,256]
    dis_model.add(Conv2D(256, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    # [-1,8*8*256]
    dis_model.add(Flatten())
    dis_model.add(Dense(4096))
    dis_model.add(LeakyReLU(alpha=0.2))

    # [-1,1]
    dis_model.add(Dense(1))
    dis_model.add(Activation('sigmoid'))

    dis_model.summary()

    return dis_model


def build_adversarial_model(gen_model, dis_model):
    model = Sequential()
    model.add(gen_model)
    dis_model.trainable = False
    model.add(dis_model)
    return model

def encoder(x,n_f):
    x=Conv2D(n_f,(3,3),padding="same")(x)
    x=BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Activation("tanh")(x)

    x = Conv2D(n_f, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    #x = MaxPooling2D((2, 2), padding="same")(x)
    x = Activation("tanh")(x)

def decoder(x,e,n_f):
    x=UpSampling2D((2,2))(x)
    x=np.concatenate(axis=1)([x,e])
    x=Conv2D(n_f,(3,3),padding="same")(x)
    x=BatchNormalization()(x)
    x = Activation("tanh")(x)

    #x = UpSampling2D((2, 2))(x)
    x = Conv2D(n_f, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("tanh")(x)


def load_images(image_paths, image_shape):
    images = None

    for i, image_path in enumerate(image_paths):
        try:
            print(i)
            #입력 이미지 자체가 흑백 이미지(생성한 가짜 이미지)
            loaded_image = cv2.imread(os.path.join(data_dir, image_path))
            loaded_image = cv2.resize(loaded_image, (256, 256))
            #엣지 추출
            #loaded_image = cv2.Canny(loaded_image, 50, 240)
            #흑백 이미지(grayscale)
            #loaded_image=cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)

            # loaded_image = cv2.filter2D(loaded_image, -1, kernel_sharpen_3)
            # loaded_image = cv2.inRange(loaded_image, lowerBound, upperBound)
            # loaded_image = image.load_img(os.path.join(data_dir, image_path), target_size=image_shape)
            loaded_image = image.img_to_array(loaded_image)
            loaded_image = np.expand_dims(loaded_image, axis=0)

            if i == 0:
                print(loaded_image)
                print(loaded_image.shape)
                cv2.imwrite("./photo.png", loaded_image[0])
            loaded_image /= 255.0

            # image에 tensor로 이어 붙이기
            if images is None:
                images = loaded_image

            else:
                images = np.concatenate([images, loaded_image], axis=0)
                # labels = np.concatenate([labels, edge_image], axis=0)
                # cv2.imwrite("./photo_{}.jpeg".format(i),loaded_image[0])
                # cv2.imwrite("./label_{}.jpeg".format(i), edge_image[0])
                # save_rgb_img(loaded_image[0], "./{}.jpeg".format(i))
        except Exception as e:
            print("Error:", i, e)

    return images


def load_data(path):
    epower_list = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        meta = csv.reader(f)
        for epower in meta:
            epower_list.append(epower)
    return epower_list


def load_img_path(dirpath):
    path = []
    images = glob.glob("{}/*.png".format(dirpath))
    for iidx, img in enumerate(images):
        path.append(img)

    return path


def write_log(writer, name, value, batch_no):
    with writer.as_default():
        tf.summary.scalar(name, value, step=batch_no)
        writer.flush()

    # summary = tf.Summary()
    # summary_value = summary.value.add()
    # summary_value.simple_value = value
    # summary_value.tag = name
    # writer.add_summary(summary,batch_no)


if __name__ == '__main__':
    # 파라미터 초기화
    data_dir = "C:\\Users\\ailab5\\PycharmProjects\\GANTest\\venv\\results"
    epochs = 1000
    batch_size = 3
    image_shape = (256, 256, 1)
    dis_learning_rate = 0.0001
    gen_learning_rate = 0.0001
    dis_momentum = 0.5
    gen_momentum = 0.5
    dis_nesterov = True
    gen_nesterov = True

    # data 준비
    imgpath = load_img_path(data_dir)
    emlist = load_images(imgpath, (image_shape[0], image_shape[1]))

    # 라벨 smoothing
    real_labels = np.ones((batch_size, 1), dtype=np.float32) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32) * 0.1

    #U-net 최적기 컴파일
    gen_optimizer = Adam(lr=gen_learning_rate, beta_1=0.9)
    gen_model = build_generator()
    gen_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    # 생성기 훈련
    for epoch in range(epochs):
        gen_losses = []
        number_of_batches = int(len(emlist) / batch_size)

        for index in range(number_of_batches):
            images_batch = emlist[index * batch_size:(index + 1) * batch_size]
            z_noise = np.random.normal(-1., 1., size=(batch_size, z_shape))
            g_loss = gen_model.train_on_batch(z_noise, y_real)

            gen_losses.append(g_loss)

        print("Epoch: ",epoch," d_loss:", d_loss," g_loss:", g_loss)

        z_noise = np.random.normal(-1., 1., size=(batch_size, z_shape))
        gen_images = gen_model.predict_on_batch(z_noise)
        #print(gen_images[0]*255.0)
        cv2.imwrite("C:/Users/ailab5/PycharmProjects/GANTest/venv/results/{}.png".format(epoch), gen_images[0]*255.)

    # 생성기, 판별기 가중치 저장
    try:
        gen_model.save("generator_model.h5")
        dis_model.save("generator_model.h5")

    except Exception as e:
        print("Eror:", e)
