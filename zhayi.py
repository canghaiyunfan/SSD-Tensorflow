import tensorflow as tf
import os
filename = r"F:\code\SSD-Tensorflow\VOC2007\test\JPEGImages\000001.jpg"
f=os.path.abspath(filename)
print(f)
#image_data = tf.gfile.FastGFile(filename, 'rb').read()