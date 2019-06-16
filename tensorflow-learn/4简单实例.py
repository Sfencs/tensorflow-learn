import tensorflow as tf
import numpy as np

# 生成100个随机点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

# 构建线性模型,逐渐接近100个随机点
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

# 误差平方再平均
loss = tf.reduce_mean(tf.square(y_data-y))

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)

# 最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(400):
        sess.run(train)
        print(sess.run([k,b]))










