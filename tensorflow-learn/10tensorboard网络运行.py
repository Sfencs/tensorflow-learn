import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 显示函数
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)



# 载入数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
# 每个批次大小
batch_size = 100
# 批次数量
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,10])

# 简单的神经网络
# 权值与偏置
with tf.name_scope('nertural_network'):
    with tf.name_scope('W'):
        W = tf.Variable(tf.zeros([784,10]))
        variable_summaries(W)
    with tf.name_scope('b'):
        b = tf.Variable(tf.zeros([10]))
        variable_summaries(b)
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(tf.matmul(x,W)+b)

with tf.name_scope('train'):
    # 二次代价函数
    # loss = tf.reduce_mean(tf.square(y-prediction))
    # 交叉熵
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
    # 梯度下降
    train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.name_scope('predict_result'):
    # 结果放在一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy',accuracy)


# 合并指标
merged = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(init)
    # 写入文件
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(80):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})

        writer.add_summary(summary,epoch)
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('次数'+str(epoch)+' : '+str(acc))














