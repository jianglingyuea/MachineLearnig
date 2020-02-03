import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import Sequential, layers
from    PIL import Image
from    matplotlib import pyplot as plt
h_dim = 20
batchsz = 512
lr = 1e-3


(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# we do not need label
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)

class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        # Encoders
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])

        # Decoders
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])


    def call(self, inputs, training=None):
        # [b, 784] => [b, 10]
        h = self.encoder(inputs)
        # [b, 10] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat



model = AE()
model.build(input_shape=(None, 784))
model.summary()
tf.math.log
optimizer = tf.optimizers.Adam(lr=lr)
for epoch in range(100):
    for step, x in enumerate(train_db):
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            x_rec_logits = model(x)
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_mean(rec_loss)
        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step % 100 == 0:
            print(epoch, step, float(rec_loss))
            # 重建图片，从测试集采样一张图片
            x = next(iter(test_db))
            logits = model(tf.reshape(x, [-1, 784]))  # 打平并送入自编码器
            x_hat = tf.sigmoid(logits)  # 将输出转换为像素值
            # 恢复为 28x28,[b, 784] => [b, 28, 28]
            x_hat = tf.reshape(x_hat, [-1, 28, 28])
            # 输入的前 50 张+重建的前 50 张图片合并， [b, 28, 28] => [2b, 28, 28]
            x_concat = tf.concat([x[:50], x_hat[:50]], axis=0)
            x_concat = x_concat.numpy() * 255.  # 恢复为 0~255 范围
            x_concat = x_concat.astype(np.uint8)
            save_images(x_concat, 'ae_images/rec_epoch_%d.png' % epoch)