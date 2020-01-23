import tensorflow as tf
x = tf.random.normal([2,5,5,3])
w = tf.random.normal([3,3,3,4])
out = tf.nn.conv2d(x, w, strides=1, padding=[[0,0],[0,0],[0,0],[0,0]])
print(out)
#padding=[[0,0],[上,下],[左,右],[0,0]]
out = tf.nn.conv2d(x, w, strides=1, padding=[[0,0],[1,1],[1,1],[0,0]])
print(out)
out = tf.nn.conv2d(x, w, strides=1, padding='SAME')
print(out)