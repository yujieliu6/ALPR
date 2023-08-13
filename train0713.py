import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'SimHei'
# The size of the images should be consistent: ensure that each image is of the same dimensions, i.e., 32 pixels in width and 40 pixels in height.
width0 = 32
height0 = 40
image_size = 1280
loops = 100 # Number of training iterations, indicating that the neural network will undergo xx iterations of training.
classes0 = 41  # The number of image classes, i.e., the images should belong to one of 41 categories. In the end, the probability for each category should be obtained.

# The set of license plate numbers to be recognized consists of 41 classes of outputs. These classes include digits 0-9, English letters A-Z (excluding I and O), and some Chinese characters.
characters0 = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Training dataset ID 0-9
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',  # Training dataset ID 10-21  (excluding I)
              'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # Training dataset ID 22-33 (excluding O)
              '京', '闽', '粤', '苏', '沪', '浙','津'  # Training dataset ID 34-40
               )

x = tf.placeholder(tf.float32, shape=[None, image_size])
y = tf.placeholder(tf.float32, shape=[None, classes0])
x_images = tf.reshape(x, [-1, width0, height0, 1])

W_con0 = tf.Variable(tf.random_normal([8, 8, 1, 16], stddev=0.1), name='W_con1')
b_con0 = tf.Variable(tf.constant(0.1, shape=[16]), name='b_con1')
jj_con0 = tf.nn.conv2d(x_images, W_con0, strides=[1, 1, 1, 1], padding="SAME")
jh_con0 = tf.nn.relu(jj_con0 + b_con0)
ch_con0 = tf.nn.max_pool(jh_con0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

W_con1 = tf.Variable(tf.random_normal([5, 5, 16, 32], stddev=0.1), name='W_con1')
b_con1 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_con2')
jj_con1 = tf.nn.conv2d(ch_con0, W_con1, strides=[1, 1, 1, 1], padding="SAME")
jh_con1 = tf.nn.relu(jj_con1 + b_con1)
ch_con1 = tf.nn.max_pool(jh_con1, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="SAME")

h_fc0_flat = tf.reshape(ch_con1, [-1, 16 * 20 * 32])
W_fc0 = tf.Variable(tf.random_normal([16 * 20 * 32, 512], stddev=0.1), name="W_fc1")
b_fc0 = tf.Variable(tf.constant(0.1, shape=[512]), name="b_fc1")
h_fc0 = tf.nn.relu(tf.matmul(h_fc0_flat, W_fc0) + b_fc0)

rate = tf.placeholder(tf.float32)
h_fc0_drop = tf.nn.dropout(h_fc0, rate)


W_fc1 = tf.Variable(tf.random_normal([512, classes0], stddev=0.1), name="W_fc2")
b_fc1 = tf.Variable(tf.constant(0.1, shape=[classes0]), name="b_fc2")
y_con = tf.matmul(h_fc0_drop, W_fc1) + b_fc1

cross = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_con))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross)

correct = tf.equal(tf.argmax(y_con, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

def prepare_image_data():
    input_count = 0
    for i in range(0, classes0):
        dir = './train_images/training-set/%s/' % str(i)
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                input_count += 1
    input_images = np.array([[0] * image_size for i in range(input_count)])
    input_labels = np.array([[0] * classes0 for i in range(input_count)])

    index = 0
    for i in range(0, classes0):
        dir = './train_images/training-set/%s/' % str(i)
        for rt, dirs, files in os.walk(dir):#rt: The current directory path being traversed.dirs: A list of subdirectories in the current directory.files: A list of file names in the current directory.
            for filename in files:
                filename = dir + filename
                img = Image.open(filename).convert('L')  # Convert the image to a grayscale image
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
    for i in range(0, classes0):
        dir = './train_images/validation-set/%s/' % str(i)  # validation-set,"i" is the classification label.
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                val_count += 1
    val_images = np.array([[0] * image_size for i in range(val_count)])
    val_labels = np.array([[0] * classes0 for i in range(val_count)])

    index = 0
    for i in range(0, classes0):
        dir = './train_images/validation-set/%s/' % str(i)  # "i" is the classification label.
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        try:
                            if img.getpixel((w, h)) > 230:
                              val_images[index][w + h * width] = 0
                            else:
                                val_images[index][w + h * width] = 1
                        except TypeError:
                            pass  # Skip the error and continue with the subsequent code
                val_labels[index][i] = 1
                index += 1
    return input_images, input_labels, input_count, val_images, val_labels, val_count


# Record the accuracy trend for each class.
class_accuracy_rates = [[] for _ in range(classes0)]
training_iterations = []
accuracy_rates = []
# Define the variable to hold the highest accuracy and the corresponding number of trainings
best_accuracy = 0.0
best_accuracy_iteration = -1
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    input_images, input_labels, input_count, val_images, val_labels, val_count = prepare_image_data()
    batch_size = 60
    batches_count = int(input_count / batch_size)
    res = input_count % batch_size
    initial_accry = sess.run(accuracy, feed_dict={x: val_images,
                                                  y: val_labels,
                                                  rate: 1.0})
    print("Initial Accuracy: %0.5f%%" % (initial_accry * 100))
    for i in range(loops):
        for n in range(batches_count):
            sess.run(train_step, feed_dict={x: input_images[n * batch_size:(n + 1) * batch_size],
                                            y: input_labels[n * batch_size:(n + 1) * batch_size],
                                            rate: 0.5})
        if res > 0:
            start_index = batches_count * batch_size
            sess.run(train_step, feed_dict={x: input_images[start_index:input_count - 1],
                                            y: input_labels[start_index:input_count - 1],
                                            rate: 0.5})

        if i % 1 == 0:

            accry = sess.run(accuracy, feed_dict={x: val_images,
                                                  y: val_labels,
                                                  rate: 1.0})
            print("Training : %d , Accuracy Rate: %0.5f%%" % (i, accry * 100))

            training_iterations.append(i)
            accuracy_rates.append(accry)
            #  Record the accuracy trend for each class.
            for class_index in range(classes0):
                class_batch_images = val_images[val_labels[:, class_index] == 1]
                class_batch_labels = val_labels[val_labels[:, class_index] == 1]
                class_accry = sess.run(accuracy, feed_dict={x: class_batch_images,
                                                            y: class_batch_labels,
                                                            rate: 1.0})
                class_accuracy_rates[class_index].append(class_accry)
                # Save the highest accuracy model
                if accry > best_accuracy:
                    best_accuracy = accry
                    best_accuracy_iteration = i
                    saver.save(sess, "./best_model.ckpt")
    print("Initial Accuracy: %0.5f%%" % (initial_accry * 100))
    print("Average Accuracy: %0.5f%%" % (sum(accuracy_rates) / len(accuracy_rates) * 100))
    # save model
    saver.save(sess, "./save_model/letter_digits_model.ckpt")

# Print the highest accuracy and corresponding training times
print("Best Accuracy: %0.5f%%" % (best_accuracy * 100))
print("Achieved in Training Iteration: %d" % best_accuracy_iteration)
# Calculate average accuracy
average_accuracy = sum(accuracy_rates) / len(accuracy_rates)
plt.plot(training_iterations, accuracy_rates, linestyle='-', color='b', label='Accuracy Rate')
plt.xlabel('Training Iterations')
plt.ylabel('Accuracy Rate (%)')
plt.title('Training Progress')
plt.grid(True)
plt.legend(loc='best')
plt.show()
# Plot the accuracy trend for each class
for class_index, class_name in enumerate(characters0):
    class_accry = class_accuracy_rates[class_index][-1]  # Obtain the final accuracy for each class.
    if class_accry < average_accuracy:
        plt.plot(training_iterations, class_accuracy_rates[class_index], label=class_name)

plt.xlabel('Training Iterations')
plt.ylabel('Accuracy Rate')
plt.title('Classes with Accuracy Lower than Average')
plt.legend(loc='best')
plt.grid(True)  # Add grid lines to the plot

# Annotate each data point for classes with accuracy lower than average
for class_index, class_name in enumerate(characters0):
    class_accry = class_accuracy_rates[class_index][-1]
    if class_accry < average_accuracy:
        plt.annotate(f'{class_accry:.2%}', (training_iterations[-1], class_accry), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()

#Reference:
#1：SOBE_rrr. (2022, September 29). [Implementing license plate recognition using CNN (Convolutional Neural Network) in TensorFlow. Complete Python project code attached]. CSDN. URL: [https://blog.csdn.net/SOBE_rrr/article/details/119924079?app_version=5.15.5&code=app_1562916241&csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22119924079%22%2C%22source%22%3A%22SOBE_rrr%22%7D&uLinkId=usr1mkqgl919blen&utm_source=app]






