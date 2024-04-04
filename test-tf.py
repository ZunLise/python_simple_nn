import  tensorflow as tf
from keras import backend as K

hello = tf.constant('Hello, TensorFlow')
sess = tf.Session()
print(sess.run(hello))

K.tensorflow_backend._get_available_gpus()