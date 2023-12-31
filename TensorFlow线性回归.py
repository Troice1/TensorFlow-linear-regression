import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

# 生成数据
Train_X = np.linspace(-10, 10, 100)
Train_Y = np.arange(100)
plt.figure(figsize=(9, 5), dpi=100)

# 绘制原始数据，加入随机噪声干扰
plt.plot(Train_X, Train_Y + np.random.randn(*Train_X.shape) * 5, "r.", label='Original data')
plt.show()

# tf图的输入
tf.compat.v1.disable_eager_execution()
X = tf.compat.v1.placeholder('float')
Y = tf.compat.v1.placeholder('float')

# 设置模型的权重与偏置
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# 构造一个线性模型
pred = tf.add(tf.multiply(X, W), b)

# 损失函数设置为均方误差
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * Train_X.shape[0])

# 梯度下降
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cost)
# 初始化变量
init = tf.compat.v1.global_variables_initializer()
print(init)
# 开始训练
with tf.compat.v1.Session() as sess:
    sess.run(init)

    for epoch in range(1000):  # 训练1000次
        for x, y in zip(Train_X, Train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # 现实每50轮迭代的结果
        if (epoch + 1) % 50 == 0:
            c = sess.run(cost, feed_dict={X: Train_X, Y: Train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(W), "b=", sess.run(b))

    print("学习完毕")
    train_cost = sess.run(cost, feed_dict={X: Train_X, Y:Train_Y})
    print("new=", train_cost, "W=", sess.run(W), "b=", sess.run(b))


# 绘制训练后的图
    plt.figure(figsize=(9, 5), dpi=100)
    plt.plot(Train_X, Train_Y + np.random.randn(*Train_X.shape) * 10, "r.", label='Original data')
    plt.plot(Train_X, sess.run(W) * Train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
