#!/usr/bin/python3
#!/usr/bin/python3
#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import time
import os
import sys
import pickle
import random


class_num = 10
image_size = 32
img_channels = 3
iterations = 250  #迭代周期，总数据/batch_size，200*250=50000
batch_size = 200
total_epoch = 200
dropout_rate = 0.5
log_save_path = './simple_logs'
model_save_path = './model/'

def download_data():  #下载
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    fname = './CAFIR-10_data/cifar-10-python.tar.gz'
    fpath = './' + dirname

    download = False
    if os.path.exists(fpath) or os.path.isfile(fname):
        download = False
        print("DataSet already exist!")
    else:
        download = True
    if download:
        print('Downloading data from', origin)
        import urllib.request
        import tarfile

        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = min(int(count*block_size*100/total_size),100)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                            (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        urllib.request.urlretrieve(origin, fname, reporthook)
        print('Download finished. Start extract!', origin)
        if fname.endswith("tar.gz"):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        elif fname.endswith("tar"):
            tar = tarfile.open(fname, "r:")
            tar.extractall()
            tar.close()

#源数据解压
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#源数据处理
def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels

#源数据处理加载
def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels

#源数据整合过程
def prepare_data():
    print("======Loading data======")
    download_data()
    data_dir = './cifar-10-batches-py'
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + '/batches.meta')

    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data(['test_batch'], data_dir, label_count)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels


#初始化结构

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev = 0.01)#初始值，生成一个截断的正太分布
#     return tf.Variable(initial)

#初始化b 偏执值
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#卷积层，用官方conv2d库，2d代表二维
def conv2d (x, W):
    #x input tensor of shape "[batch, in_height, in_width, in_channels]"
    #W filter滤波器（扫描那个），"[filter_height, filer_width, in_channels, out_channels]"
    #strides[0]=1,[3]=1:第一和第四个值都要为1。strides[1]:x方向的步长，strides[2]:y方向的步长
    #padding: "SAME","VALID"两种
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = "SAME")

#池化层 可用max_pool, men_pool, stochastic_pooling随机池化
def max_pool_3x3(x):
    #ksize [1,x,y,1],窗口大小x*y去扫描，第一第四个值一样为1
    return tf.nn.max_pool(x, ksize = [1,3,3,1], strides = [1,2,2,1], padding = "SAME")

#BarchNorm
def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)


def learning_rate_schedule(epoch_num): #按区间控制learning rate，可能这种效果比一次一次降低学习率要好，需要文献支撑
    if epoch_num < 100:
        return 0.01
    elif epoch_num < 150:
        return 0.001
    else:
        return 0.0001


# 测试集数据取值测试
def run_testing(sess, ep):
    acc = 0.0
    loss = 0.0
    pre_index = 0
    add = 1000
    for it in range(10):
        batch_x = test_x[pre_index:pre_index + add]  # 索引，一个batch一次取1000行，也就是1000张测试集照片，循环10次，回到0——1000行取
        batch_y = test_y[pre_index:pre_index + add]
        pre_index = pre_index + add  # 累加实现索引区间递增

        loss_, acc_ = sess.run([cross_entropy, accuracy],
                               feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: False})

        loss += loss_ / 10.0  # 结果的权重只占测试集跑出来的10%，累加，因为跑一次数据只有总数的10%，这样的结果应该合理。
        acc += acc_ / 10.0


        # 为tensorboard结构图用的
    summary = tf.Summary(value=[tf.Summary.Value(tag="Loss_1", simple_value=loss),
                                tf.Summary.Value(tag="Acc_1", simple_value=acc)])

    return acc, loss, summary


# 主体结构
if __name__ == '__main__':

    train_x, train_y, test_x, test_y = prepare_data()  # 调用上面处理好的数据
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)  # 用于BatchNorm

    # 卷积网络层

    W_conv1 = tf.get_variable('conv1', shape=[5, 5, 3, 32],initializer = tf.truncated_normal_initializer(stddev = 0.0001))
    b_conv1 = bias_variable([32])  # 每一个卷积核一个偏执值
    output = tf.nn.relu(batch_norm(conv2d(x, W_conv1) + b_conv1))  # BatchNor设置好了，启动设为true即可，fals为不BatchNor
    output = max_pool_3x3(output)

    W_conv2 = tf.get_variable('conv2', shape=[5, 5, 32, 64], initializer = tf.truncated_normal_initializer(stddev = 0.01))
    b_conv2 = bias_variable([64])  # 每一个卷积核一个偏执值
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2) + b_conv2))  # BatchNor设置好了，启动设为true即可，fals为不BatchNor
    output = max_pool_3x3(output)

    W_conv3 = tf.get_variable('conv3', shape=[5, 5, 64, 64], initializer = tf.truncated_normal_initializer(stddev = 0.01))
    b_conv3 = bias_variable([64])  # 每一个卷积核一个偏执值
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3) + b_conv3))  # BatchNor设置好了，启动设为true即可，fals为不BatchNor
    output = max_pool_3x3(output)

    # 全连接层
    output = tf.reshape(output, [-1, 4 * 4 * 64])

    W_fc1 = tf.get_variable('fc1', shape=[4 * 4 * 64, 1024], initializer = tf.truncated_normal_initializer(stddev = 0.1))
    b_fc1 = bias_variable([1024])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc1) + b_fc1))
    output = tf.nn.dropout(output, keep_prob)

    W_fc2 = tf.get_variable('fc2', shape=[1024, 128], initializer = tf.truncated_normal_initializer(stddev = 0.1))
    b_fc2 = bias_variable([128])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc2) + b_fc2))
    output = tf.nn.dropout(output, keep_prob)

    W_fc3 = tf.get_variable('fc3', shape=[128, 10], initializer = tf.truncated_normal_initializer(stddev = 0.1))
    b_fc3 = bias_variable([10])
    output = tf.nn.softmax(batch_norm(tf.matmul(output, W_fc3) + b_fc3))

    # loss函数
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output, name="Loss"))  # 损失函数交叉熵
    tf.summary.scalar("Loss", cross_entropy)  # tensorboard记录值

    # l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()]) #正则化

    # 优化器
    # train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay) #用正则化情况下，用这条代码
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32, name="Acc"))  # 正确率
    tf.summary.scalar("Acc", accuracy)  # tensorboard记录值

    # initial an saver to save model  开始启动
    saver = tf.train.Saver()

    merged = tf.summary.merge_all()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_save_path, sess.graph)  # tensorboard流程图储存路径
        # 写入tensorboard
        train_writer = tf.summary.FileWriter("logs/AAtrain_BatchNormm_lr0.01_change_w_stable", sess.graph)
        test_writer = tf.summary.FileWriter("logs/AAtest_BatchNormm_lr0.01_change_w_stable", sess.graph)

        # epoch = 164
        # make sure [bath_size * iteration = data_set_number]

        for ep in range(1, total_epoch + 1):
            lr = learning_rate_schedule(ep)
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            start_time = time.time()

            print("\n epoch %d/%d:" % (ep, total_epoch))  # 1/200形式

            for it in range(1, iterations + 1):  # 训练集区域
                batch_x = train_x[pre_index:pre_index + batch_size]  # 索引取训练数据
                batch_y = train_y[pre_index:pre_index + batch_size]

                # batch_x = data_augmentation(batch_x) #将取出的训练数据拿去增强

                _, batch_loss = sess.run([train_step, cross_entropy],  # 启动训练，batch_loss获得loss值
                                         feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout_rate,
                                                    learning_rate: lr, train_flag: True})  # 开启关闭BatchNorm，dropout设置

                batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})

                train_loss += batch_loss  # 累加，因为你正在最小化的目标函数的价值。这个值可能是正数或负数，具体取决于具体的目标函数。
                train_acc += batch_acc  # 累加
                pre_index += batch_size  # 这个累加用于更新取值范围

                if it == iterations:
                    train_loss /= iterations  # 求平均值，更合理
                    train_acc /= iterations

                    loss_, acc_ = sess.run([cross_entropy, accuracy],
                                           feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})

                    summary = sess.run(merged, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})
                    train_writer.add_summary(summary, ep)

                    # 为tensorboard结构图用的
                    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                                                      tf.Summary.Value(tag="train_accuracy", simple_value=train_acc)])

                    val_acc, val_loss, test_summary = run_testing(sess, ep)  # 传到测试集函数区域做计算取值

                    test_writer.add_summary(test_summary, ep)

                    summary_writer.add_summary(train_summary, ep)
                    summary_writer.add_summary(test_summary, ep)
                    summary_writer.flush()

                    print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
                          "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
                          % (it, iterations, int(time.time() - start_time), train_loss, train_acc, val_loss, val_acc))

                else:
                    print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f"
                          % (it, iterations, train_loss / it, train_acc / it), end='\r')  # 一个周期不跑完，不跑训练集结果

            tf.summary.merge_all()
        save_path = saver.save(sess, model_save_path)  # model储存路径
        print("Model saved in file: %s" % save_path)
