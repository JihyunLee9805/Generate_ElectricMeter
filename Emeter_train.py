import os
import csv
import time
from datetime import datetime
import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras import Input, Model
from keras.applications import InceptionResNetV2
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Reshape, concatenate, LeakyReLU, Lambda, \
     Activation, UpSampling2D, Dropout

import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing import image
from scipy.io import loadmat
#import skimage.io
#import cv2

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def BEncoder():
    input_layer = Input(shape=(256, 256, 3))

    # [-1, 128,128,64]
    enc = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(input_layer)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # [-1, 64, 64, 128]
    enc = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # [-1, 32, 32, 256]
    enc = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # [-1, 16, 16,512]
    enc = Conv2D(filters=512, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # [-1, 8, 8,512]
    enc = Conv2D(filters=512, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # [-1, 4, 4,512]
    enc = Conv2D(filters=512, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    enc = Flatten()(enc)

    # [-1,4096]
    enc = Dense(4096)(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # [-1, 1000]
    enc = Dense(1000)(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    model = Model(inputs=[input_layer], outputs=[enc])
    return model

def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def BGenerator():
    latent_dims=1000
    num_class=10000

    input_z_noise=Input(shape=(latent_dims,))
    input_label=Input(shape=(num_class,))

    x=concatenate([input_z_noise,input_label]) #x shape=[-1,10100]

    #1 FC = [-1, 4096]
    x = Dense(4069, input_dim=latent_dims + num_class)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    #2 FC = [-1, 256*8*8]
    x = Dense(512 * 4 * 4)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    # [-1, 4, 4, 256]
    x = Reshape((4, 4, 512))(x)

    #1 UpSampling = [-1, 8, 8, 512]
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    #2 UpSampling = [-1, 16, 16, 512]
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=512, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # 3 UpSampling = [-1, 32, 32, 256]
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # 4 UpSampling = [-1, 64,64, 128]
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # 4 UpSampling = [-1, 128,128, 64]
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # 7 UpSampling = [-1,256, 256, 3]
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=3, kernel_size=5, padding='same')(x)
    x = Activation('tanh')(x)

    model = Model(inputs=[input_z_noise, input_label], outputs=[x])
    return model

def expand_label_input(x):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=1)
    x = K.tile(x, [1, 128, 128, 1])
    return x

def BDiscriminator():
    input_shape=(256,256,3)
    label_shape=(10000,)
    image_input=Input(shape=input_shape)
    label_input=Input(shape=label_shape)

    #1 Conv = [-1,128,128,64]
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(image_input)
    x = LeakyReLU(alpha=0.2)(x)

    #2 Conv = [-1, 128,128,10128]
    label_input1 = Lambda(expand_label_input)(label_input)
    print(x.shape)
    print(label_input1.shape)
    x = concatenate([x, label_input1], axis=3)

    #3 Conv = [-1,64, 64,128]
    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    #4 Conv = [-1, 32,32,256]
    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    #5 Conv = [-1, 16,16,512]
    x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # 6 Conv = [-1, 8,8,512]
    x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)

    #FC [-1, 1]
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[image_input, label_input], outputs=[x])
    return model

def load_images(image_paths, image_shape):
    images = None

    for i, image_path in enumerate(image_paths):
        try:
            print(image_path)
            loaded_image = image.load_img(os.path.join(data_dir, image_path), target_size=image_shape)
            loaded_image = image.img_to_array(loaded_image)
            loaded_image = np.expand_dims(loaded_image, axis=0)
            loaded_image /= 255.

            # image에 tensor로 이어 붙이기
            if images is None:
                images = loaded_image
            else:
                images = np.concatenate([images, loaded_image], axis=0)
                #save_rgb_img(loaded_image[0], "./{}.jpeg".format(i))
        except Exception as e:
            print("Error:", i, e)

    return images

def load_img_path(dirpath):
    path=[]
    label=[]
    #dirs=["meter1","meter2"]
    #ddirs=["0000","0100","0200","0300","0400","0500","0600","0700","0800","0900","1000"]
    dirs = ["meter1"]
    ddirs=["0000"]

    for i,type in enumerate(dirs):
        p=os.path.join(dirpath,type)
        for idx, img_path in enumerate(ddirs):
            ipath=os.path.join(p,img_path)
            images=glob.glob("{}/*.jpg".format(ipath))
            for iidx, img in enumerate(images):
                path.append(img)
                label.append(img[len(img)-11:len(img)-7])

    return path,label


def load_data(path):
    epower_list=[]
    with open(path,'r',encoding='utf-8-sig') as f:
        meta=csv.reader(f)
        for epower in meta:
            epower_list.append(epower)
    return epower_list

def save_rgb_img(img, path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow((img * 255).astype(np.uint8))
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)
    plt.close()

def write_log(writer,name, value, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    writer.add_summary(summary,batch_no)


if __name__ == '__main__':
    #파라미터 초기화
    data_dir="D:\meter_dataset\meter_images_2160"
    label_dir="/Users/jihyun/Documents/4-1/외부활동/인턴논문및특허/EMETER/epower.csv"
    epochs=1000
    batch_size=3
    image_shape=(128, 128, 3)
    z_shape=1000

    #신경망 최적화
    dis_optimizer = Adam(lr=0.005, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    gen_optimizer = Adam(lr=0.005, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    adversarial_optimizer = Adam(lr=0.05, beta_1=0.5, beta_2=0.999, epsilon=10e-8)

    #생성기 + 판별기 신경망 컴파일
    discriminator=BDiscriminator()
    discriminator.trainable = False
    discriminator.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)
    generator = BGenerator()
    generator.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    input_z_noise = Input(shape=(1000,))
    input_label = Input(shape=(10000,))
    recons_images = generator([input_z_noise, input_label])
    valid = discriminator([recons_images, input_label])

    #적대모델 생성
    adversarial_model = Model(inputs=[input_z_noise, input_label], outputs=[valid])
    adversarial_model.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)


    #data 준비
    imgpath,eplist=load_img_path(data_dir)
    print(eplist)
    y=to_categorical(int(eplist[0]),10000)
    emlist= load_images(imgpath, (image_shape[0], image_shape[1]))

    #라벨 smoothing
    real_labels = np.ones((batch_size, 1), dtype=np.float32) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32) * 0.1

    session = tf.compat.v1.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/', session.graph)

    #생성기 훈련
    for epoch in range(epochs):
        print("Epoch: ",epoch)

        gen_losses=[]
        dis_losses=[]
        number_of_batches = int(len(emlist) / batch_size)
        print("Number of batches:", number_of_batches, len(emlist))

        for index in range(number_of_batches):

            images_batch = emlist[index * batch_size:(index + 1) * batch_size]
            images_batch = images_batch / 127.5 - 1.0
            images_batch = images_batch.astype(np.float32)

            #y_batch=[batch_size, 10000]
            y_batch = y[index * batch_size:(index + 1) * batch_size]
            #z_noise=[batch_size,1000]
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

            #초기 가짜 이미지 생성
            print(z_noise.shape,y_batch.shape)
            init_recon_img=generator.predict_on_batch([z_noise, y_batch])

            #판별자 학습

            d_loss_real = discriminator.train_on_batch([images_batch, y_batch], real_labels)
            d_loss_fake = discriminator.train_on_batch([init_recon_img, y_batch], fake_labels)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            print("d_loss:{}".format(d_loss))

            #잠재벡터 생성
            z_noise2 = np.random.normal(0, 1, size=(batch_size, z_shape))
            random_labels = np.random.randint(0, 10000, batch_size).reshape(-1, 1)
            random_labels = to_categorical(random_labels, 10000)

            #적대적 신경망 훈련
            g_loss = adversarial_model.train_on_batch([z_noise2, random_labels], [1] * batch_size)

            print("g_loss:{}".format(g_loss))

            gen_losses.append(g_loss)
            dis_losses.append(d_loss)

        write_log(writer,'g_loss', np.mean(gen_losses), epoch)
        write_log(writer,'d_loss', np.mean(dis_losses), epoch)

        if epoch % 2 == 0:
            images_batch = emlist[0:batch_size]
            images_batch = images_batch / 127.5 - 1.0
            images_batch = images_batch.astype(np.float32)

            y_batch = y[0:batch_size]
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

            gen_images = generator.predict_on_batch([z_noise, y_batch])

            for i, img in enumerate(gen_images[:1]):
                save_rgb_img(img, path="./results/img_{}_{}.png".format(epoch, i))

    #생성기, 판별기 가중치 저장
    try:
        generator.save_weights("generator.h5")
        discriminator.save_weights("discriminator.h5")
    except Exception as e:
        print("Eror:", e)



    #인코더 훈련
    encoder=BEncoder()
    encoder.compile(loss=euclidean_distance_loss, optimizer='adam')


    try:
        generator.load_weights("generator.h5")
    except Exception as e:
        print("Error:", e)

    # [5,8192]
    z_i = np.random.normal(0, 1, size=(5, z_shape))

    # [-1, 10000,1]
    y = np.random.randint(low=0, high=10000, size=(5,), dtype=np.int64)
    num_classes = len(set(y))
    y = np.reshape(np.array(y), [len(y), 1])
    print(y.shape)
    y = to_categorical(y, num_classes=10000)

    for epoch in range(epochs):
        print("Epoch:", epoch)

        encoder_losses = []

        number_of_batches = int(z_i.shape[0] / batch_size)
        for index in range(number_of_batches):
            print("Batch:", index + 1)

            z_batch = z_i[index * batch_size:(index + 1) * batch_size]
            y_batch = y[index * batch_size:(index + 1) * batch_size]

            generated_images = generator.predict_on_batch([z_batch, y_batch])

            #인코더 훈련
            encoder_loss = encoder.train_on_batch(generated_images, z_batch)
            print("Encoder loss:", encoder_loss)

            encoder_losses.append(encoder_loss)

        write_log(writer, "encoder_loss", np.mean(encoder_losses), epoch)


    #인코더 가중치 저장
    encoder.save_weights("encoder.h5")

    #테스트할 이미지 가져오기
    test_image = image.load_img(data_dir + '8.jpeg', target_size=(1024, 1024, 3))
    encoder.predict(test_image,verbose=3)




