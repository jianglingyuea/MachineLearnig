import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
x = tf.constant([2., 1., 0.1])
layer = layers.Softmax(axis=-1)
print(layer(x))
network = Sequential(
    [
        layers.Dense(3, activation=None),
        layers.ReLU(),
        layers.Dense(2, activation=None), # 全连接层
        layers.ReLU() #激活函数层
    ]
)
x = tf.random.normal([4,3])
print(network(x))
layers_num = 2
network = Sequential([])
for _ in range(layers_num):
    network.add(layers.Dense(3))
    network.add(layers.ReLU())
network.build(input_shape=(None,4))
print(network.summary())
for p in network.trainable_variables:
    print(p.name, p.shape)