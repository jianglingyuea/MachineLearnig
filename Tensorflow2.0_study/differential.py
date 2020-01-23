import tensorflow as tf
x = tf.constant(1.)
w1 = tf.constant(2.)
b1 = tf.constant(1.)
w2 = tf.constant(2.)
b2 = tf.constant(1.)
with tf.GradientTape(persistent=True) as tape:
    tape.watch([w1, b1, w2, b2])
    y1 = x * w1 + b1
    y2 = y1 * w2 + b2
dy2_dy1 = tape.gradient(y2, [y1])[0]
dy1_dw1 = tape.gradient(y1, [w1])[0]
dy2_dw1 = tape.gradient(y2, [w1])[0]

print(dy2_dy1 * dy1_dw1)
print(dy2_dw1)
import  numpy as np
from    mpl_toolkits.mplot3d import Axes3D
from    matplotlib import pyplot as plt



def himmelblau(x):
    # himmelblau函数实现
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
# 生成x-y平面采样网格点，方便可视化
X, Y = np.meshgrid(x, y)
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y]) # 计算网格点上的函数值
# 绘制himmelblau函数曲面
fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

x = tf.constant([-2.,2.])
for step in range(1000):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)
    grads = tape.gradient(y, [x])[0]
    x -= grads * 0.001
    if step % 20 == 19:
        print('step {}: x = {}, f(x) = {}'
              .format(step, x.numpy(), y.numpy()))
