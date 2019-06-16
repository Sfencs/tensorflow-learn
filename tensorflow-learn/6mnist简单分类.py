import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
# 每个批次大小
batch_size = 100
# 批次数量
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

# 简单的神经网络
# 权值
W = tf.Variable(tf.zeros([784,10]))
# 偏置
b = tf.Variable(tf.zeros([10]))

prediction = tf.nn.softmax(tf.matmul(x,W)+b)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# 交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# 梯度下降
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for eoch in range(80):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('次数'+str(eoch)+' : '+str(acc))













