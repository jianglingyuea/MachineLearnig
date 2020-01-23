
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, losses

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
#print(dataset_path)
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment="\t",
                      sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())
print(dataset.head())

# 统计空白数据,并清除
print("before:", dataset.isna().sum())
dataset = dataset.dropna()
print("after:", dataset.isna().sum())
print(dataset)
#%%

origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


#%% 统计数据
sns.pairplot(train_dataset[["Cylinders", "Displacement", "Weight", "MPG"]],
diag_kind="kde")

train_stats = train_dataset.describe()
print(train_stats)
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# 标准化数据
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
#%%

print(normed_train_data.shape,train_labels.shape)
print(normed_test_data.shape, test_labels.shape)
# p = normed_train_data.describe().transpose()
# print(p)

class Network(keras.Model):
    # 回归网络
    def __init__(self):
        super(Network, self).__init__()
        # 创建3个全连接层
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        # 依次通过3个全连接层
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

model = Network()
model.build(input_shape=(None, 9))
model.summary()
optimizer = tf.keras.optimizers.RMSprop(0.001)
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
train_db = train_db.shuffle(100).batch(32)
print(train_db)
train_mae_losses = []
test_mae_losses = []
for epoch in range(200):
    for step, (x, y) in enumerate(train_db):

        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.reduce_mean(losses.MSE(y, out))
            mae_loss = tf.reduce_mean(losses.MAE(y, out))

        if step % 10 == 0:
            print(epoch, step, float(loss))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_mae_losses.append(float(mae_loss))
    out = model(tf.constant(normed_test_data.values))
    test_mae_losses.append(tf.reduce_mean(losses.MAE(test_labels, out)))

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.plot(train_mae_losses, label='Train')

plt.plot(test_mae_losses, label='Test')
plt.legend()

# plt.ylim([0,10])
plt.legend()
plt.savefig('auto.svg')
plt.show()