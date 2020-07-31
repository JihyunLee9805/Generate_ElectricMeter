import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import glob
import io
import math
import time
import csv
import cv2

import keras.backend as K

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
import os
import glob


def build_generator():
    gen_model = Sequential()

    # [-1, 4096]
    gen_model.add(Dense(input_dim=300, output_dim=4096))
    gen_model.add(ReLU())
    #gen_model.add(Dropout(0.1))

    # [-1, 256*8*8]
    gen_model.add(Dense(256 * 8 * 8))
    gen_model.add(BatchNormalization(momentum=0.9))
    gen_model.add(ReLU())

    #[-1, 8, 8, 256]
    gen_model.add(Reshape((8, 8, 256), input_shape=(256 * 8 * 8,)))
    gen_model.add(Conv2DTranspose(256, (3, 3), padding='same'))
    gen_model.add(BatchNormalization(momentum=0.9))
    gen_model.add(ReLU())
   # gen_model.add(Dropout(0.4))

    # [-1,16,16,128]
    gen_model.add(UpSampling2D(size=(2, 2)))
    gen_model.add(Conv2DTranspose(128, (3, 3), padding='same'))
    gen_model.add(BatchNormalization(momentum=0.9))
    gen_model.add(ReLU())
    #gen_model.add(Dropout(0.1))

    # [-1,32,32,64]
    gen_model.add(UpSampling2D(size=(2, 2)))
    gen_model.add(Conv2DTranspose(64, (3, 3),padding='same'))
    gen_model.add(BatchNormalization(momentum=0.9))
    gen_model.add(ReLU())
    #gen_model.add(Dropout(0.4))

    # [-1,64,64,32]
    gen_model.add(UpSampling2D(size=(2, 2)))
    gen_model.add(Conv2DTranspose(32,(3, 3),  padding='same'))
    gen_model.add(BatchNormalization(momentum=0.9))
    gen_model.add(ReLU())

    #[-1,128,128,16]
    gen_model.add(UpSampling2D(size=(2, 2)))
    gen_model.add(Conv2DTranspose(16, (3, 3),  padding='same'))
    gen_model.add(BatchNormalization(momentum=0.9))
    gen_model.add(ReLU())
    #gen_model.add(Dropout(0.1))

    # [-1,256,256,8]
    gen_model.add(UpSampling2D(size=(2, 2)))
    gen_model.add(Conv2DTranspose(8,(3, 3),  padding='same'))
    gen_model.add(BatchNormalization(momentum=0.9))
    gen_model.add(ReLU())

    # [-1,256,256,1]
    #gen_model.add(UpSampling2D(size=(2, 2)))
    gen_model.add(Conv2DTranspose(1, (3, 3), padding='same'))
    gen_model.add(BatchNormalization(momentum=0.9))
    gen_model.add(Activation('tanh'))
    gen_model.summary()

    return gen_model


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


def load_images(image_paths, image_shape):
    images = None

    lowerBound = np.array([0, 0, 0])
    upperBound = np.array([65, 65, 65])
    kernel_sharpen_3 = np.array(
        [[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]])

    for i, image_path in enumerate(image_paths):
        try:
            print(i)
            loaded_image = cv2.imread(os.path.join(data_dir, image_path))

            loaded_image = cv2.resize(loaded_image, (256, 256))
            #엣지 추출
            #loaded_image = cv2.Canny(loaded_image, 50, 240)
            #흑백 이미지(grayscale)
            loaded_image=cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)

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
<<<<<<< HEAD:dcgan_edge256256_v2.py
    #dirs=["meter1","meter2"]
    #ddirs=["0000","0100","0200","0300","0400","0500","0600","0700","0800","0900","1000"]
    #ddirs = ["0000", "0100", "0200", "0300", "0400", "0500"]
    dirs = ["meter1"]
    ddirs = ["0000"]

    for i, type in enumerate(dirs):
        p = os.path.join(dirpath, type)
        print(p)
        for idx, img_path in enumerate(ddirs):
            ipath = os.path.join(p, img_path)
            images = glob.glob("{}/*.jpg".format(ipath))
            for iidx, img in enumerate(images):
                path.append(img)
=======
    dirs=["meter1","meter2"]
    #ddirs=["0000","0100","0200","0300","0400","0500","0600","0700","0800","0900","1000"]
    ddir1 = ["0000", "0100", "0200", "0300", "0400", "0500"]
    ddir2 = ["0500","0600","0700","0800","0900","1000"]
    #dirs = ["meter1"]
    #ddirs = ["0000"]

    for i, type in enumerate(dirs):
        p = os.path.join(dirpath, type)
        if i==0:
            for idx, img_path in enumerate(ddir1):
                ipath = os.path.join(p, img_path)
                images = glob.glob("{}/*.jpg".format(ipath))
                for iidx, img in enumerate(images):
                    path.append(img)
        else:
            for idx, img_path in enumerate(ddir2):
                ipath = os.path.join(p, img_path)
                images = glob.glob("{}/*.jpg".format(ipath))
                for iidx, img in enumerate(images):
                    path.append(img)
>>>>>>> 79093df88a3d120082e715dc5f4d5011768ad0b4:dcgan_edge256256.py
    return path


def denormalize(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8)


def normalize(img):
    return (img - 127.5) / 127.5


def visualize_rgb(img):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")
    plt.show()


def save_rgb_img(img, path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow((img * 255).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)
    plt.close()


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
<<<<<<< HEAD:dcgan_edge256256_v2.py
    data_dir = "C:\\Users\\jihyun\\PycharmProjects\\GANTest\\venv\\repo\\Generate_ElectricMeter\\meter_dataset\\meter_images_2160"

    epochs = 10000
=======
    data_dir = "D:\meter_dataset\meter_images_2160"
    label_dir = "/Users/jihyun/Documents/4-1/외부활동/인턴논문및특허/EMETER/epower.csv"
    epochs = 300
>>>>>>> 79093df88a3d120082e715dc5f4d5011768ad0b4:dcgan_edge256256.py
    batch_size = 3
    image_shape = (256, 256, 1)
    z_shape = 300
    dis_learning_rate = 0.0001
    gen_learning_rate = 0.0001
    dis_momentum = 0.5
    gen_momentum = 0.5
    dis_nesterov = True
    gen_nesterov = True

    # 신경망 최적화
    dis_optimizer = Adam(lr=dis_learning_rate, beta_1=0.9)
    # ad_optimizer = SGD(lr=gen_learning_rate, momentum=gen_momentum, nesterov=gen_nesterov)
    # dis_optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    gen_optimizer = Adam(lr=gen_learning_rate, beta_1=0.9)

    # 생성기 + 판별기 신경망 컴파일
    dis_model = build_discriminator()
    dis_model.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    gen_model = build_generator()
    gen_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    adversarial_model = build_adversarial_model(gen_model, dis_model)
    adversarial_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
    print("AD")

    imgpath = load_img_path(data_dir)

    # data 준비
    emlist = load_images(imgpath, (image_shape[0], image_shape[1]))

    # 라벨 smoothing
    real_labels = np.ones((batch_size, 1), dtype=np.float32) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32) * 0.1

    # writer = tf.summary.create_file_writer('./logs')

    # 생성기 훈련
    for epoch in range(epochs):
        gen_losses = []
        dis_losses = []
        number_of_batches = int(len(emlist) / batch_size)

        for index in range(number_of_batches):
            images_batch = emlist[index * batch_size:(index + 1) * batch_size]

            #판별기 먼저 학습
            z_noise = np.random.normal(-1., 1., size=(batch_size, z_shape))
            generated_images = gen_model.predict_on_batch(z_noise)
            dis_model.trainable = True
            #판별기 입력에 인공적 노이즈 추가
            y_real = np.ones((batch_size,)) * 0.9
            y_fake = np.zeros((batch_size,)) * 0.1
            # 판별자 학습할때 입력은 하나의 종류만 넣고 학습, 섞어서 사용 x
            dis_loss_real = dis_model.train_on_batch(images_batch, y_real)
            dis_loss_fake = dis_model.train_on_batch(generated_images, y_fake)

            d_loss = (dis_loss_real + dis_loss_fake)


            dis_model.trainable = False
            #z_noise = np.random.randint(0, high=0 + 1, size=(batch_size, z_shape))
            z_noise = np.random.normal(-1., 1., size=(batch_size, z_shape))
            g_loss = adversarial_model.train_on_batch(z_noise, y_real)

            dis_losses.append(d_loss)
            gen_losses.append(g_loss)

            # write_log(writer, 'g_loss', np.mean(gen_losses), epoch)
            # write_log(writer, 'd_loss', np.mean(dis_losses), epoch)

        print("Epoch: ",epoch," d_loss:", d_loss," g_loss:", g_loss)

        #z_noise = np.random.randint(0, high=0 + 1, size=(batch_size, z_shape))
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
