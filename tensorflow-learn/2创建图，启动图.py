import tensorflow as tf


# 创建常量op
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])

# 创建矩阵乘法op
product = tf.matmul(m1,m2)
print(product)
# 创建会话，启动默认图
sess = tf.Session()
# 在会话中执行
result = sess.run(product)
print(result)
sess.close()

# 使用上下文管理器自动关闭会话
with tf.Session() as sess:
    result = sess.run(product)
    print(result)




