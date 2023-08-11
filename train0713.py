# 1.图片的大小要统一：保证每一张图片都是一样的大  宽度*高度=32*40
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image

WIDTH = 32 # 设定图像的宽度为32个像素
HEIGHT = 40 # 设定图像的长度为40个像素
IMAGESIZE = 1280  # 计算图像的总大小，即宽度乘以长度，此处为32*40=1280个像素。
interations = 100  # 训练次数，表示神经网络将进行xx次迭代训练。
NUM_CLASSES = 41  # 图像的分类数量，即图像应属于41个类别中的一种。最后要得出每一种的概率

# 要识别的车牌号码的集合 41类输出。其中包括数字0-9，英文字母A-Z（不包括I和O），以及一些汉字
LETTER_NUM = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 训练集编号0-9
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',  # 训练集编号10-21 没有I
              'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # 训练集编号22-33 没有O
              '京', '闽', '粤', '苏', '沪', '浙','津'  # 训练集编号34-40
              )

x = tf.placeholder(tf.float32, shape=[None, IMAGESIZE])
y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
x_imgs = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])

W_con1 = tf.Variable(tf.random_normal([8, 8, 1, 16], stddev=0.1),name='W_con1')
b_con1 = tf.Variable(tf.constant(0.1, shape=[16]), name='b_con1')
jj_con1 = tf.nn.conv2d(x_imgs, W_con1, strides=[1, 1, 1, 1], padding="SAME")
jh_con1 = tf.nn.relu(jj_con1 + b_con1)
ch_con1 = tf.nn.max_pool(jh_con1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

W_con2 = tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.1), name='W_con1')
b_con2 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_con2')
jj_con2 = tf.nn.conv2d(ch_con1, W_con2, strides=[1, 1, 1, 1], padding="SAME")
jh_con2 = tf.nn.relu(jj_con2 + b_con2)
ch_con2 = tf.nn.max_pool(jh_con2, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],padding="SAME")

h_fc1_flat = tf.reshape(ch_con2, [-1, 16 * 20 * 32])
W_fc1 = tf.Variable(tf.random_normal([16 * 20 * 32, 512], stddev=0.1), name="W_fc1")  # 权重
b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name="b_fc1")  # 生成偏置 512个0.1
h_fc1 = tf.nn.relu(tf.matmul(h_fc1_flat, W_fc1) + b_fc1)

rate = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, rate)


W_fc2 = tf.Variable(tf.random_normal([512, NUM_CLASSES], stddev=0.1), name="W_fc2")  # 权重
b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b_fc2")  # 生成偏置
y_con = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_con))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross)  # AdamOptimizer设置了学习率

correct = tf.equal(tf.argmax(y_con, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # 准确率

init = tf.global_variables_initializer()  # 变量初始化

def picRead_pre():
    input_count = 0
    for i in range(0, NUM_CLASSES):
        dir = './train_images/training-set/%s/' % str(i)
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                input_count += 1
    input_images = np.array([[0] * IMAGESIZE for i in range(input_count)])
    input_labels = np.array([[0] * NUM_CLASSES for i in range(input_count)])

    index = 0
    for i in range(0, NUM_CLASSES):
        dir = './train_images/training-set/%s/' % str(i)
        for rt, dirs, files in os.walk(dir):#rt：正在遍历的当前目录路径。dirs：当前目录中的子目录列表。files：当前目录中的文件名列表。
            for filename in files:
                filename = dir + filename
                img = Image.open(filename).convert('L')  # 将图像转换为灰度图像
                #img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        if img.getpixel((w, h)) > 230:
                            input_images[index][w + h * width] = 0
                        else:
                            input_images[index][w + h * width] = 1
                input_labels[index][i] = 1
                index += 1

    val_count = 0
    for i in range(0, NUM_CLASSES):
        dir = './train_images/validation-set/%s/' % str(i)  # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                val_count += 1
    val_images = np.array([[0] * IMAGESIZE for i in range(val_count)])
    val_labels = np.array([[0] * NUM_CLASSES for i in range(val_count)])

    index = 0
    for i in range(0, NUM_CLASSES):
        dir = './train_images/validation-set/%s/' % str(i)  # 这里可以改成你自S己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        if img.getpixel((w, h)) > 230:
                            val_images[index][w + h * width] = 0
                        else:
                            val_images[index][w + h * width] = 1
                val_labels[index][i] = 1
                index += 1
    return input_images, input_labels, input_count, val_images, val_labels, val_count

with tf.Session() as sess:
    sess.run(init)
    input_images, input_labels, input_count, val_images, val_labels, val_count = picRead_pre()
    batch_size = 60
    batches_count = int(input_count / batch_size)
    res = input_count % batch_size

    for i in range(interations):
        for n in range(batches_count):
            sess.run(train_step, feed_dict={x: input_images[n * batch_size:(n + 1) * batch_size],
                                            y: input_labels[n * batch_size:(n + 1) * batch_size],
                                            rate: 0.5})
        if res > 0:  # 如果图片剩余
            start_index = batches_count * batch_size
            sess.run(train_step, feed_dict={x: input_images[start_index:input_count - 1],
                                            y: input_labels[start_index:input_count - 1],
                                            rate: 0.5})

        if i % 1 == 0:
            # 得到准确率
            accry = sess.run(accuracy, feed_dict={x: val_images,
                                                  y: val_labels,
                                                  rate: 1.0})
            print("Training : %d , Accuracy Rate: %0.5f%%" % (i, accry * 100))
            # Print weights and biases
    print("第1层卷积层_权重：", sess.run(W_con1), "偏置：", sess.run(b_con1))
    print("第2层卷积层_权重：", sess.run(W_con2), "偏置：", sess.run(b_con2))
    print("第1个全连接层权重：", sess.run(W_fc1), "偏置：", sess.run(b_fc1))
    print("第2个全连接层权重：", sess.run(W_fc2), "偏置：", sess.run(b_fc2))

    print("训练完成")
    # 模型的保存
    saver = tf.train.Saver()
    saver.save(sess, "./save_model/letter_digits_model.ckpt")

