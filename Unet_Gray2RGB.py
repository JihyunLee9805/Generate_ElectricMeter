import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

from keras import *
from keras.layers import *
from keras.activations import *
from keras.optimizers import *
from matplotlib import pyplot as plt
from keras.initializers import RandomNormal, Zeros
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing import image

import glob
import io
import os
import cv2


def residual_block(feature, dropout=False):
    x = Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(feature)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return Add()([feature, x])

def build_model(n_block=5):
    image_size = 224
    input_channel = 1
    output_channel = 3
    input = Input(shape=(image_size, image_size, input_channel))

    #Encoder-downsampling
    x = Conv2D(224, kernel_size=7, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(input)  # use reflection padding instead
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = Conv2D(200, kernel_size=3, strides=2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(n_block):
        x = residual_block(x)

    #Decoder-upsampling
    x = Conv2DTranspose(200, kernel_size=3, strides=2, padding='same',
                        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(224, kernel_size=3, strides=2, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(output_channel, kernel_size=7, padding='same', kernel_initializer=RandomNormal(
        mean=0.0, stddev=0.02), bias_initializer=Zeros())(x)  # use reflection padding instead
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    model = Model(inputs=input, outputs=x)
    model.summary()

    return model


def unnormalize(image):
    image = (image*255)
    return image.astype('uint8')

def save_images(generator, samples):
    ab_values = generator.predict(samples[0:1])
    genImg=unnormalize(ab_values[0])
    cv2.imwrite("./generated_img.png",genImg)

def train_network(x_data,y_data,epochs=100, batch_size=3, save_interval=5):
    input_shape = (224, 224, 1)
    output_shape = (224, 224, 2)
    print(y_data.shape)

    autoEncoder = build_model()
    autoEncoder.compile(optimizer='adam', loss='mse', metrics=['mse','acc'])
    history = autoEncoder.fit(x_data,y_data,validation_split=0.1,epochs=epochs,batch_size=3,)
    autoEncoder.save('./Unet_autoencoder.hdf5')
    print("MODEL SAVED")

    return autoEncoder,history

def load_img_path(dirpath):
    path = []
    dirs=["meter1","meter2"]
    #ddirs=["0000","0100","0200","0300","0400","0500","0600","0700","0800","0900","1000"]
    ddirs = ["0000", "0100"]
    #dirs = ["meter1"]
    #ddirs = ["0000"]

    for i,type in enumerate(dirs):
        p=os.path.join(dirpath,type)
        for idx, img_path in enumerate(ddirs):
            ipath=os.path.join(p,img_path)
            images=glob.glob("{}/*.jpg".format(ipath))
            for iidx, img in enumerate(images):
                path.append(img)

    return path

def load_images(image_paths):
    images = None
    RGBImages=None

    for i, image_path in enumerate(image_paths):
        try:
            print(i)
            loaded_image = cv2.imread(os.path.join(data_dir, image_path))

            #엣지 추출
            #loaded_image = cv2.Canny(loaded_image, 50, 240)
            # 컬러 이미지(rgb)
            rgb_loaded = cv2.resize(loaded_image, (224, 224))
            #흑백 이미지(grayscale)
            loaded_image = cv2.resize(loaded_image, (224, 224))
            loaded_image=cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)

            loaded_image = image.img_to_array(loaded_image)
            loaded_image = np.expand_dims(loaded_image, axis=0)
            rgb_loaded= image.img_to_array(rgb_loaded)
            rgb_loaded = np.expand_dims(rgb_loaded, axis=0)

            if i == 0:
                print(loaded_image)
                print(loaded_image.shape)
                cv2.imwrite("./photo.png", loaded_image[0])
                cv2.imwrite("./photo_color.png", rgb_loaded[0])
            loaded_image /= 255.0
            rgb_loaded/=255.0

            # image에 tensor로 이어 붙이기
            if images is None:
                images = loaded_image
                RGBImages=rgb_loaded

            else:
                images = np.concatenate([images, loaded_image], axis=0)
                RGBImages = np.concatenate([RGBImages,rgb_loaded], axis=0)
        except Exception as e:
            print("Error:", i, e)

    return images,RGBImages

if __name__ == '__main__':
    # 파라미터 초기화
    data_dir = "D:\meter_dataset\meter_images_2160"
    image_shape=(224,224,1)

    imgpath = load_img_path(data_dir)
    grayImg,rgbImg = load_images(imgpath)
    model,histroy=train_network(grayImg,rgbImg)
    save_images(model,grayImg)