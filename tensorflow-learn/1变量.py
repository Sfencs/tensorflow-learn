import tensorflow as tf

# 变量
x = tf.Variable([1,2])
# 常量
a = tf.constant([3,3])
# 减法op
sub = tf.subtract(x,a)
# 加法op
add = tf.add(x,sub)

# 全局变量初始化
init = tf.global_variables_initializer()

# 会话
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))



state = tf.Variable(0,name='counter') # 可以起名字
new_value = tf.add(state,1)
# 赋值
update = tf.assign(state,new_value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        print(sess.run(update))


