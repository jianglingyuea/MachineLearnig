import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
x = tf.constant([[1,2],[2,1]])
print(tf.reshape(x,[4, -1]))
# x = tf.constant([1,2,3.3])
# print(x)
# print(x.numpy())
# a = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])
# print(a)
# a = tf.constant(True)
# print(
#     a
# )
# if a == True:
#     print(1)
# else:
#     print(0)
# b = tf.constant(1.6, dtype=tf.float16)
# print(b)
# b = tf.cast(b, tf.float32)
# print(b)
# a = tf.constant([-1, 0, 1, 2])
# a = tf.cast(a, tf.bool)
# print(a)
# a = tf.constant([-1, 0, 1, 2])
# aa = tf.Variable(a)
# print(aa.name, aa.trainable)
# print(tf.convert_to_tensor(np.array([[1,2.],[3,4]])))
# a = tf.zeros([2,3])
# print(tf.ones_like(a))
# print(tf.fill([2,2,2], -1))
# print(tf.random.uniform([2,2], minval=10, maxval=100, dtype=tf.int32))
# print(tf.range(1, 10, delta=2))
# out = tf.random.uniform([4,10]) #随机模拟网络输出
# print(out)
# y = tf.constant([2,3,2,0]) # 随机构造样本真实标签
# y = tf.one_hot(y, depth=10) # one-hot 编码
# print("y= ", y)
# loss = tf.keras.losses.mse(y, out) # 计算每个样本的 MSE
# loss = tf.reduce_mean(loss) # 平均 MSE
# print(loss)
# z = tf.random.normal([4,2])
# print("z=", z)
# b = tf.ones([2]) # 模拟偏置向量
# print("b=", b)
# z = z + b # 累加偏置
# print(z)
# fc = layers.Dense(3,kernel_initializer=tf.keras.initializers.Zeros()) # 创建一层 Wx+b，输出节点为 3
# # 通过 build 函数创建 W,b 张量，输入节点为 4
# fc.build(input_shape=(2,4))
# print(fc.bias)
# print(fc.kernel)
# (x_train,y_train),(x_test,y_test)=keras.datasets.imdb.load_data(num_words=10000)
# x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=80)
# print(x_train.shape)
# embedding=layers.Embedding(10000, 100)
# out = embedding(x_train)
# print(out.shape)
#
# # 创建 32x32 的彩色图片输入，个数为 4
# x = tf.random.normal([4,32,32,3])
# # 创建卷积神经网络
# layer = layers.Conv2D(16,kernel_size=3)
# out = layer(x) # 前向计算
# print(out.shape)
#
# print(x[0:2,...,1:].shape)
# print(x[2:,...].shape)
# print(x[0:2,::-2,::-2,1:].shape)



# x = tf.random.uniform([4,4],maxval=10,dtype=tf.int32)
# x = tf.expand_dims(x,axis=0)
# print(x)
#
# x = tf.range(4)
# x = tf.reshape(x, [2, 2])
# print("x=",x)
# x = tf.tile(x, multiples=[1,3])
# print("xtile=",x)
# x = tf.tile(x, multiples=[2,1])
# print("xtile=", x)
# A = tf.random.normal([32,32])
# tf.broadcast_to(A, [2,32,32,32])
# a = tf.random.normal([4,3,23,32])
# b = tf.random.normal([4,3,32,2])
# print(a@b)


a = tf.random.normal([1,3,4])
b = tf.random.normal([2,3,4])
c = tf.concat([a,b],axis=0)
print(c.shape)
x = tf.ones([2,2])
print(tf.norm(x,1))

out = tf.random.normal([100,10])
out = tf.nn.softmax(out, axis=1) # 输出转换为概率
pred = tf.argmax(out, axis=1)
print(pred)
a = tf.constant([1,2,3])
b = tf.constant([4,5,6])
print(a.shape)
c = tf.stack([a,b],axis=1)
print(c)

a = tf.ones([3, 3])
b = tf.zeros([3, 3])
cond = tf.constant([[True, False, False], [False, True, True], [True, True, True]])
print(tf.where(cond, a ,b))

indices = tf.constant([[4], [3], [1], [7]])
updates = tf. constant([1, 2, 3 ,4])
print(tf.scatter_nd(indices, updates, [8]))