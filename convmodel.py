# -*- coding: utf-8 -*-

# Sample code to use string producer.
import os
from filecmp import cmp
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# RESIZING IMAGES DATA SET




# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3     # How many classes do you have ?
batch_size = 5      # Minimum 5. It is futile a batch size less than 5

"""
car_images = tf.train.match_filenames_once("IMGR/car/*")
motorbike_images = tf.train.match_filenames_once("IMGR/motorbike/*")
airplane_images = tf.train.match_filenames_once("IMGR/airplane/*")


filename_queue0 = tf.train.string_input_producer(car_images, shuffle=False)
filename_queue1 = tf.train.string_input_producer(motorbike_images, shuffle=False)
filename_queue2 = tf.train.string_input_producer(airplane_images, shuffle=False)

reader0 = tf.WholeFileReader()
reader1 = tf.WholeFileReader()
reader2 = tf.WholeFileReader()

key0, file_image0 = reader0.read(filename_queue0)
key1, file_image1 = reader1.read(filename_queue1)
key2, file_image2 = reader2.read(filename_queue2)

image0, label0 = tf.image.decode_jpeg(file_image0), [1., 0., 0.]  # key0 car
image0 = tf.reshape(image0, [80, 140, 1])

image1, label1 = tf.image.decode_jpeg(file_image1), [0., 1., 0.]  # key1 motorbike
image1 = tf.reshape(image1, [80, 140, 1])

image2, label2 = tf.image.decode_jpeg(file_image2), [0., 0., 1.]  # key 2 airplane
image2 = tf.reshape(image2, [80, 140, 1])

image0 = tf.to_float(image0) / 256. - 0.5
image1 = tf.to_float(image1) / 256. - 0.5
image2 = tf.to_float(image2) / 256. - 0.5
"""

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), 3) # [one_hot(float(i), num_classes)]
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch

""""    
example_batch0, label_batch0 = tf.train.shuffle_batch([image0, label0], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

example_batch1, label_batch1 = tf.train.shuffle_batch([image1, label1], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)


example_batch2, label_batch2 = tf.train.shuffle_batch([image2, label2], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)


example_batch = tf.concat(values=[example_batch0, example_batch1, example_batch2], axis=0)
label_batch = tf.concat(values=[label_batch0, label_batch1, label_batch2], axis=0)
"""
def convolutionalModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        #
        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(["IMGR/car/*", "IMGR/motorbike/*", "IMGR/airplane/*"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["IMGR/carVal/*", "IMGR/motorbikeVal/*", "IMGR/airVal/*"], batch_size=batch_size)

example_batch_train_predicted = convolutionalModel(example_batch_train, reuse=False)
example_batch_valid_predicted = convolutionalModel(example_batch_valid, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train , dtype=tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, dtype=tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

train_errors = []
validation_errors = []
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())


    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for _ in range(2000):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(label_batch_train))
            sess.run(label_batch_valid)
            train_error = sess.run(cost)
            val_error = sess.run(cost_valid)
            validation_errors.append(val_error)
            train_errors.append(train_error)
            print("Error:", train_error)

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
            
    coord.request_stop()
    coord.join(threads)

# x_axis_errors =  list(range(1, len(train_errors) + 1))
x_axis_errors = [each * 20 for each in range(1, len(train_errors)+1)]

def getFigure(plotName, xAxisName, yAxisName, fileName, x, y):
    fig = plt.figure()
    plt.plot(x, y)
    fig.suptitle(plotName, fontsize=20)
    plt.xlabel(xAxisName)
    plt.ylabel(yAxisName)
    fig.savefig(fileName+'.jpg')
    plt.show()


getFigure("Training Errors", "Epochs", "Error", "training_errorp", x_axis_errors, train_errors)
getFigure("Validation Errors", "Epochs", "Error", "val_errorp", x_axis_errors, validation_errors)