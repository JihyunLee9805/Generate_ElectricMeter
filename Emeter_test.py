import os
import Emeter_train as em
from keras_preprocessing import image
import numpy as np

generator=em.BGenerator()
discriminator=em.BDiscriminator()
encoder=em.BEncoder()

#generator.load_weights("./generator.h5",True)
#discriminator.load_weights("./discriminator.h5",True)
#encoder.load_weights("./encoder.h5",True)

data_dir="/Users/jihyun/Documents/4-1/외부활동/인턴논문및특허/EMETER/data/"
loaded_image = image.load_img(data_dir+'8.jpeg', target_size=(1024,1024,3))
loaded_image = image.img_to_array(loaded_image)
loaded_image = np.expand_dims(loaded_image, axis=0)

em.save_rgb_img(loaded_image[0],"./test.jpeg")
#encoder.predict(loaded_image,verbose=3)