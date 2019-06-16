import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成200个随机点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]

# 干扰项
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data)+noise

# 定义两个占位符
# 行不确定，列为一列
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

# 中间层神经网络
# 权值
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
# 偏置
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 输出层网络
Weights_L2 = tf.Variable(tf.random.normal([10,1]))
# 偏置
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))

# 梯度下降训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 训练
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    # 预测
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()



