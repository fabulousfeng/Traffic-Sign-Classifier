### Load the images and plot them here.
### Feel free to use as many code cells as needed.
%matplotlib inline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import cv2
import numpy as np

img = cv2.imread('00649.ppm')
plt.imshow(img)
images =  cv2.resize(img, (32,32))
images = np.reshape(images, (1,32,32,3))

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
import pandas as pd

images = (images)/255.0 - 0.5
sign_names = pd.read_csv('signnames.csv', sep=',')

predict = tf.argmax(logits, 1)

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./lenet.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('.'))
    output = sess.run(predict, feed_dict={x: images, keep_prob: 1.0})
    print(sign_names['SignName'][output])
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(dic[output[0]])
