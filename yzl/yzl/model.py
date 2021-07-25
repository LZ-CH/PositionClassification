import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import time


def read_data(path='./data/train/'):
    '''
    读入原始数据
    '''
    labels = ['000', '001', '002', '003', '004']
    data2 = []
    label2 = []
    length = []
    for label in labels:
        path2 = path + label
        # print(path2)
        files = os.listdir(path2)
        for file in files:
            data2.append(np.load(path2+'/'+file))
            label2.append(int(label))
            length.append(data2[-1].shape[2])
            # print(data2[-1].shape)
    return data2, label2, length


def data2io(x, y, sl, l=32):
    '''
    将原始数据转化为大小(m,32,102)的矩阵
    同时把数据归一化为(-1,1)之间的实数
    '''
    m1 = len(x)
    xe = []
    ye = []
    if l > min(sl):
        l = min(sl)
    for i in range(m1):
        m2 = x[i].shape[2]
        for j in range(m2-l):
            xei = x[i][:, :, j:j + l, :, :]
            xe.append(xei)
            ye.append(y[i])
    xe = np.array(xe)
    xe = np.squeeze(a=xe, axis=1)
    ye = np.array(ye)
    xe = np.swapaxes(a=xe, axis1=2, axis2=4)
    xe = np.reshape(a=xe, newshape=(xe.shape[0], -1, xe.shape[-1]))
    xe = np.swapaxes(a=xe, axis1=1, axis2=2)
    ye = tf.keras.utils.to_categorical(ye, num_classes=5)

    xe = xe / (1e-6 + np.max(a=np.abs(xe), axis=-1)[:, :, np.newaxis])

    return xe, ye


def create_placeholder(sl=32, hs=102):
    x = tf.placeholder(dtype=tf.float32, shape=(None, sl, hs))
    y = tf.placeholder(dtype=tf.float32, shape=(None, 5))
    return x, y


def forward(x_input):
    '''
    模型结构，与ppt中画出的完全相同
    '''
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=4, strides=1, padding='same')(x_input)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.0)(x)

    x = tf.keras.layers.Conv1D(filters=256, kernel_size=4, strides=1, padding='same')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.0)(x)

    x = tf.keras.layers.Conv1D(filters=512, kernel_size=4, strides=1, padding='same')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.0)(x)

    x = tf.keras.layers.Conv1D(filters=1024, kernel_size=4, strides=1, padding='valid')(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.0)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    x = tf.keras.layers.Dense(units=32, activation='relu')(x)
    x = tf.keras.layers.Dense(units=5, activation='relu')(x)

    x = tf.keras.layers.Softmax()(x)

    return x


def compute_cost(y_pred, y_true):
    cost = tf.keras.losses.CategoricalCrossentropy()(y_true=y_true, y_pred=y_pred)
    return cost


def get_minibatch(x, y, bs):
    m = x.shape[0]
    x_minibatch = []
    y_minibatch = []
    for i in range(m):
        a = i * bs
        b = i * bs + bs
        if b >= m:
            x_minibatch.append(x[a:m, :, :])
            y_minibatch.append(y[a:m, :])
            break
        else:
            x_minibatch.append(x[a:b, :, :])
            y_minibatch.append(y[a:b, :])
    return x_minibatch, y_minibatch


def define_opt(cost, lr=1e-3):
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    return opt


def split_y(nv, y):
    '''
    将预处理中分解的相应向量合并，以便使用平均投票计算准确率
    '''
    y_sp2 = np.split(ary=y, indices_or_sections=list(nv))[0:-1]
    y_sp3 = []
    for i in range(len(y_sp2)):
        y_sp3.append(np.sum(a=y_sp2[i], axis=0))
    y_sp3 = np.array(y_sp3)
    return y_sp3


def train(epoch, batch_size, fix_len, lr):

    x_train, y_train, sl = read_data()
    x_train, y_train = data2io(x_train, y_train, sl, l=fix_len)
    x_test, y_test, sl = read_data(path='./data/test/')
    y_test_sp = tf.keras.utils.to_categorical(np.array(y_test), num_classes=5)
    x_test, y_test = data2io(x_test, y_test, sl, l=fix_len)
    nv = np.array(sl) - fix_len
    nv2 = np.zeros(nv.shape, dtype=int)
    for i in range(nv.size):
        nv2[i] = np.sum(a=nv[0:i+1])

    train_cost = []
    test_cost = []
    train_acc = []
    test_acc = []
    epoch_cost = 0
    best_model_path = 'model//best.ckpt'
    last_model_path = 'model//last.ckpt'
    m1 = x_train.shape[0]
    m2 = y_test_sp.shape[0]
    best_test_acc = 0

    x, y = create_placeholder(sl=fix_len)
    y_pred = forward(x_input=x)
    cost = compute_cost(y_pred=y_pred, y_true=y)
    opt = define_opt(cost=cost, lr=lr)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        print('start training')
        for i in range(epoch):
            t1 = time.time()
            np.random.seed(seed=i)
            np.random.shuffle(x_train)
            np.random.seed(seed=i)
            np.random.shuffle(y_train)
            x_train_mb, y_train_mb = get_minibatch(x_train, y_train, bs=batch_size)
            for j in range(len(x_train_mb)):
                xj = x_train_mb[j]
                yj = y_train_mb[j]
                _, minibatch_cost = sess.run(fetches=[opt, cost], feed_dict={x: xj, y: yj})
                epoch_cost += minibatch_cost
            epoch_cost = epoch_cost/len(x_train_mb)
            train_cost.append(epoch_cost)

            [y_pred_train] = sess.run(fetches=[y_pred], feed_dict={x: x_train, y: y_train})
            train_acc_i = np.sum(np.equal(np.argmax(a=y_pred_train, axis=1), np.argmax(a=y_train, axis=1))) / m1
            train_acc.append(train_acc_i)

            [y_pred_test, test_cost_ep] = sess.run(fetches=[y_pred, cost], feed_dict={x: x_test, y: y_test})
            test_acc_i = np.sum(np.equal(np.argmax(a=split_y(nv2, y_pred_test), axis=1), np.argmax(a=y_test_sp, axis=1))) / m2
            test_acc.append(test_acc_i)
            test_cost.append(test_cost_ep)

            t2 = time.time()
            t = t2 - t1

            print('epoch:%04d  train_loss:%13.10f  test_loss:%13.10f  train_acc:%8.6f  test_acc:%8.6f  time:%02dm%02ds'
                  % (i, epoch_cost, test_cost_ep, train_acc_i, test_acc_i, t // 60, t % 60))


            if test_acc_i > best_test_acc:
                best_test_acc = test_acc_i
                saver = tf.train.Saver()
                saver.save(sess=sess, save_path=best_model_path)
                print('best model saved')


        train_cost = np.array(train_cost)
        test_cost = np.array(test_cost)
        train_acc = np.array(train_acc)
        test_acc = np.array(test_acc)


        saver = tf.train.Saver()
        saver.save(sess=sess, save_path=last_model_path)

        plt.plot(train_cost)
        plt.plot(test_cost)
        plt.savefig(fname='cost.svg')
        plt.cla()
        plt.plot(train_acc)
        plt.plot(test_acc)
        plt.savefig(fname='acc.svg')

        print('finish training!')

    return train_cost, test_cost, train_acc, test_acc


def evaluate(model_path, data_path='./data/test/'):
    '''
    加载预训练模型测试效果
    '''
    fix_len = 32
    x_test, y_test, sl = read_data(path=data_path)
    y_test_sp = tf.keras.utils.to_categorical(np.array(y_test), num_classes=5)
    x_test, y_test = data2io(x_test, y_test, sl, l=fix_len)
    nv = np.array(sl) - fix_len
    nv2 = np.zeros(nv.shape, dtype=int)
    m2 = y_test_sp.shape[0]
    for i in range(nv.size):
        nv2[i] = np.sum(a=nv[0:i + 1])

    _ = np.zeros(shape=(y_test.shape[0], 5))
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(model_path+'.meta')
        saver.restore(sess=sess, save_path=model_path)
        x = graph.get_tensor_by_name('Placeholder:0')
        y = graph.get_tensor_by_name('Placeholder_1:0')
        output = graph.get_tensor_by_name('softmax/Softmax:0')
        y_predict = sess.run(fetches=output, feed_dict={x: x_test, y: _})
        # names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        acc = np.sum(
            np.equal(np.argmax(a=split_y(nv2, y_predict), axis=1), np.argmax(a=y_test_sp, axis=1))) / m2
    return acc


