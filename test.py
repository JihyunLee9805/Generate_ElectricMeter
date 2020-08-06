import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import time


def load():
    return load_model("./generator_model.hdf5"),load_model("./Unet_autoencoder.hdf5")

def generate_img(gen,batch):

    z_noise = np.random.normal(-1., 1., size=(batch, 300))
    grayImg= gen.predict_on_batch(z_noise)
    print(grayImg.shape)
    images=[]
    for idx,img in enumerate(grayImg):
        img = (img * 255).astype('uint8')
        cv2.imwrite("./test/gray_{}.png".format(idx), img)
        time.sleep(5)
        images.append(img)
        print("SAVE GRAY IMG {}".format(idx))
    return images

def ToRGBImg(images,unet):
    for idx,gimg in enumerate(images):
        gimg = cv2.imread("./test/gray_{}.png".format(idx))

        # 흑백 이미지(grayscale)
        gimg = cv2.resize(gimg, (224, 224))
        gimg = cv2.cvtColor(gimg, cv2.COLOR_BGR2GRAY)

        gimg = image.img_to_array(gimg)
        gimg = np.expand_dims(gimg, axis=0)

        img = unet.predict_on_batch(gimg/255.)
        img = (img * 255).astype('uint8')
        print("TEST: ", idx, "SAVE")
        cv2.imwrite("./test/rgb_{}.png".format(idx), img[0])

gen,unet=load()
gen.summary()
unet.summary()
images=generate_img(gen,10)
ToRGBImg(images,unet)