import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model
from PIL import Image

size=(28,28)
dir_img="./"
img_name="0.png"
tlabel=0
ori_image = Image.open(dir_img+img_name)
ori_image=ori_image.convert('1')
pre_image = ori_image.resize(size, Image.ANTIALIAS)

plt.imshow(pre_image, cmap='gray_r')
plt.title("original {}".format(tlabel))
plt.show()

pre_image=np.array(pre_image).reshape((1,28,28,1))

myModel = load_model('mnistmodel.h5')

predict = myModel.predict(pre_image)
print(predict)
predict = np.argmax(predict)
print('original:', tlabel)
print('predicted:', predict)