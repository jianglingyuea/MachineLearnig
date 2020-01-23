from tensorflow.keras import Sequential
from tensorflow.keras import layers, datasets
network = Sequential(
    [
        layers.Conv2D(6, kernel_size=3, strides=1),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.ReLU(),
        layers.Conv2D(16, kernel_size=3, strides=1),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10)
    ]
)
network.build(input_shape=(None, 28, 28, 1))
# 统计网络信息
print(network.summary())
from tensorflow.keras import losses, optimizers
import tensorflow as tf
def proprocess(x,y):
    x = tf.expand_dims(x, -1)
    return x,y
(x, y), (x_test,y_test) = datasets.mnist.load_data()
# x: [0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

# x: [0~255] => [0~1.]
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(proprocess).batch(32)

val_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
val_db = val_db.shuffle(1000).map(proprocess).batch(32)
criteon = losses.CategoricalCrossentropy(from_logits=True)
optimizer = optimizers.Adam(lr=0.01)
for epoch in range(5):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = network(x, training=True)
            y_onehot = tf.one_hot(y, depth=10)
            loss = criteon(y_onehot, out)
            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))
            if step % 100 == 0:
                print("loss:", loss)


#test
correct, total = 0, 0
for x, y in val_db:
    out = network(x, training=False)
    pred = tf.argmax(out, axis=-1)
    y = tf.cast(y, tf.int64)
    correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
    total += x.shape[0]
print("test acc = ", correct / total)

