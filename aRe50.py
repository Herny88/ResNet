#!/usr/bin/python3
import collections
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
total_epoch = 350
log_save_path = './simple_logs'
model_save_path = './model/'
slim = tf.contrib.slim


def download_data():  # 下载
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
            percent = min(int(count * block_size * 100 / total_size), 100)
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


# 源数据解压
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 源数据处理
def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels


# 源数据处理加载
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


# 源数据整合过程
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


##################################################################################
# 图片随机裁剪
def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


# 图片随机左右翻转
def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


# 数据预处理-颜色处理并特征标准化，即图片白化。0，1，2为3原色通道。高级写法
# http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing
# https://www.cnblogs.com/luxiao/p/5534393.html
def data_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


# 数据增强
def data_augmentation(batch):
    batch = _random_flip_leftright(batch)  # 图片随机翻转
    batch = _random_crop(batch, [32, 32], 4)  # 随机切割
    return batch

#############################################################################################

def learning_rate_schedule(epoch_num):  # 按区间控制learning rate，可能这种效果比一次一次降低学习率要好，需要文献支撑
    if epoch_num < 160:
        return 0.1
    elif epoch_num < 240:
        return 0.01
    elif epoch_num < 320:
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
                               feed_dict={x: batch_x, y_: batch_y})

        loss += loss_ / 10.0  # 结果的权重只占测试集跑出来的10%，累加，因为跑一次数据只有总数的10%，这样的结果应该合理。
        acc += acc_ / 10.0


        # 为tensorboard结构图用的
    summary = tf.Summary(value=[tf.Summary.Value(tag="Loss_1", simple_value=loss),
                                tf.Summary.Value(tag="Acc_1", simple_value=acc)])

    return acc, loss, summary


# block
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block."""


# 采样
def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, rate=rate, padding='VALID', scope=scope)


# 堆叠
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None, store_non_strided_activations=False, outputs_collections=None):
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            block_stride = 1
            for i, unit in enumerate(block.args):
                if store_non_strided_activations and i == len(block.args) - 1:
                    block_stride = unit.get('stride', 1)
                    unit = dict(unit, stride=1)

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)
                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError('The target output_stride cannot be reached.')

                            # Collect activations at the block's end before performing subsampling.
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
            # Subsampling of the block's output activations.
            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride
            else:
                net = subsample(net, block_stride)
                current_stride *= block_stride
        if output_stride is not None and current_stride > output_stride:
            raise ValueError('The target output_stride cannot be reached.')

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net


# 超参
def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': None,  # Use fused batch norm if possible.
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc

#shortcut
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None,
                                   activation_fn=None, scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


# resNet主结构
def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    with slim.arg_scope([slim.conv2d],
                                        activation_fn=None, normalizer_fn=None):
                        # net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                        net = conv2d_same(net, 64, 7, stride=2, scope='conv1')  # 开始第一层
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = stack_blocks_dense(net, blocks, output_stride)
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points

def resnet_v2_block(scope, base_depth, num_units, stride):
    return Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }])


#50层
def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    blocks = [
    resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
    resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
    resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
    resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
    return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)


# 开始训练
if __name__ == '__main__':

    train_x, train_y, test_x, test_y = prepare_data()  # 调用上面拆包出来的原数据
    train_x, test_x = data_preprocessing(train_x, test_x)  # 白化预处理数据

    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    learning_rate = tf.placeholder(tf.float32)

    #     arg_scope=resnet_arg_scope()
    #     with slim.arg_scope(arg_scope):
    #         my_net,my_end_points=resnet_v2_50(x, num_classes = 10, is_training=True)
    #     output = my_end_points['predictions']


    with slim.arg_scope(resnet_arg_scope()):
        net, end_points = resnet_v2_50(x, 10,
                                       is_training=True,
                                       scope='resnet')
    output = end_points['predictions']

    # loss函数
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output, name="Loss"))  # 损失函数交叉熵
    tf.summary.scalar("Loss", cross_entropy)  # tensorboard记录值

    # 优化器
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy)

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
        train_writer = tf.summary.FileWriter("logs/Atrain_Re50", sess.graph)
        test_writer = tf.summary.FileWriter("logs/Atest_Re50", sess.graph)

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

                batch_x = data_augmentation(batch_x)  # 将取出的训练数据拿去增强

                _, batch_loss = sess.run([train_step, cross_entropy],  # 启动训练，batch_loss获得loss值
                                         feed_dict={x: batch_x, y_: batch_y, learning_rate: lr})

                batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})

                train_loss += batch_loss  # 累加，因为你正在最小化的目标函数的价值。这个值可能是正数或负数，具体取决于具体的目标函数。
                train_acc += batch_acc  # 累加
                pre_index += batch_size  # 这个累加用于更新取值范围

                if it == iterations:
                    train_loss /= iterations  # 求平均值，更合理
                    train_acc /= iterations

                    loss_, acc_ = sess.run([cross_entropy, accuracy],
                                           feed_dict={x: batch_x, y_: batch_y})

                    summary = sess.run(merged, feed_dict={x: batch_x, y_: batch_y})
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
