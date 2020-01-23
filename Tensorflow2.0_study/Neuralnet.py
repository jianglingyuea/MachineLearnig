import tensorflow as tf
x = tf.random.normal([4, 28*28])
from tensorflow.keras import layers
# print(h1.shape)
# print(fc.kernel)
# print(fc.trainable_variables)
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
# 输出层张量 
w4 = tf.Variable(tf.random.truncated_normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))
with tf.GradientTape() as tape:
    h1 = tf.nn.relu(x@w1 + b1)
    h2 = tf.nn.relu(h1@w2 + b2)
    h3 = tf.nn.relu(h2@w3 + b3)
    h4 = h3@w4 + b4

fc1 = layers.Dense(256, activation=tf.nn.relu) # 隐藏层 1
fc2 = layers.Dense(128, activation=tf.nn.relu) # 隐藏层 2
fc3 = layers.Dense(64, activation=tf.nn.relu) # 隐藏层 3
fc4 = layers.Dense(10, activation=None) # 输出层
x = tf.random.normal([4,28*28])
h1 = fc1(x) # 通过隐藏层 1 得到输出
h2 = fc2(h1) # 通过隐藏层 2 得到输出
h3 = fc3(h2) # 通过隐藏层 3 得到输出
h4 = fc4(h3) # 通过输出层得到网络输出


