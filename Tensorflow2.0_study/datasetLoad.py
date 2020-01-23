import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets # 导入经典数据集加载模块
import  matplotlib
from    matplotlib import pyplot as plt
# Default parameters for plots
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus']=False
lr = 5e-3
def preprocess(x, y): # 自定义的预处理函数
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28]) # 打平
    y = tf.cast(y, dtype=tf.int32) # 转成整形张量
    y = tf.one_hot(y, depth=10) # one-hot 编码
    return x,y
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000)
train_db = train_db.batch(256)
train_db = train_db.map(preprocess)
train_db = train_db.repeat(80)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(256).map(preprocess)
x,y = next(iter(train_db))
print('train sample:', x.shape, y.shape)
accs,losses = [], []
# 784 => 512
w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
# 512 => 256
w2, b2 = tf.Variable(tf.random.normal([256, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
w3, b3 = tf.Variable(tf.random.normal([256, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
# 256 => 10
w4, b4 = tf.Variable(tf.random.normal([256, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

for step, (x, y) in enumerate(train_db):
    with tf.GradientTape() as tape:
        h1 = x @ w1 + b1
        h1 = tf.nn.relu(h1)
        h2 = h1 @ w2 + b2
        h2 = tf.nn.relu(h2)
        h3 = h2 @ w3 + b3
        h3 = tf.nn.relu(h3)
        out = h3 @ w4 + b4
        loss = tf.square(y - out)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3, w4, b4])
    for p, g in zip([w1, b1, w2, b2, w3, b3, w4, b4], grads):
        p.assign_sub(lr * g)
    if step % 80 == 0:
        print(step, 'loss:', float(loss))
        losses.append(float(loss))
        total, total_correct = 0., 0
        for x, y in test_db:
            # layer1.
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            h3 = h2 @ w3 + b3
            h3 = tf.nn.relu(h3)
            out = h3 @ w4 + b4
            # [b, 10] => [b]
            pred = tf.argmax(out, axis=1)
            # convert one_hot y to number y
            y = tf.argmax(y, axis=1)
            # bool type
            correct = tf.equal(pred, y)
            # bool tensor => int tensor => numpy
            total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
            total += x.shape[0]

        print(step, 'Evaluate Acc:', total_correct / total)
        accs.append(total_correct / total)
plt.figure()
x = [i*80 for i in range(len(losses))]
plt.plot(x, losses, color='C0', marker='s', label='训练')
plt.ylabel('MSE')
plt.xlabel('Step')
plt.legend()
plt.savefig('train.svg')

plt.figure()
plt.plot(x, accs, color='C1', marker='s', label='测试')
plt.ylabel('准确率')
plt.xlabel('Step')
plt.legend()
plt.savefig('test.svg')