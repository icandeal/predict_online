"""
使用全连接神经网络训练样本
"""
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import common_config as conf
from urllib import parse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 这是默认的显示等级，显示所有信息  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error   
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error
# model本地存放位置
model_dir = os.environ['ModelLocalDir']

sess = tf.Session()
Accuracy = conf.Accuracy
step = conf.step

n_feature = int(16 * ((1 - Accuracy) * 100 // step + 1)) + 1  # 增加知识点的平均难度属性

hidden_layer = 100
x = tf.placeholder(tf.float32, [None, n_feature])
keep_prob = tf.placeholder(tf.float32)
W0 = tf.Variable(tf.truncated_normal([n_feature, hidden_layer], stddev=0.1), name="W0")
b0 = tf.Variable(tf.zeros([hidden_layer]), dtype=tf.float32, name="b0")
y0 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, W0) + b0), keep_prob)

W = tf.Variable(tf.truncated_normal([hidden_layer, 4], stddev=0.1), name="W")
b = tf.Variable(tf.zeros([4]), dtype=tf.float32, name="b")
y = tf.nn.softmax(tf.matmul(y0, W) + b)

y_value = tf.argmax(y, 1) + 1


# def do_predict(initialized):
def do_predict(environ, initialized):
    print("in module!!!!!!!!!!!")
    content_length = int(environ['CONTENT_LENGTH'])
    wsgi_input = environ['wsgi.input']
    request_body = wsgi_input.read(content_length)
    content = parse.parse_qs(str(request_body, 'utf-8'))
    params = {}
    for key in content.keys():
        value = content.get(key)[0]
        params[str(key)] = str(value)

    try:
        data = params["data"]
    except KeyError:
        return False

    # data = "9212346,280600 48:0.3333 58:0.3888 32:1.331 65:2.03 81:1.8848 28:0.2564 37:0.677 113:0.593 49:2.2939 53:0.3703"

    last_moment = time.time()

    line_list = data.split(" ")
    data_list = [(line_list[0], list(map(lambda a: (a.split(':')[0], a.split(':')[1]), line_list[1:])))]

    n_sample = len(data_list)
    print('数据加载完毕,总样本数：', n_sample, '，用时:', time.time() - last_moment)
    last_moment = time.time()

    # 整理测试样本
    test_x_array = np.zeros([n_sample, n_feature + 2], dtype="float32")
    for i in range(n_sample):
        for feature in data_list[i][1]:
            test_x_array[i][int(feature[0]) + 2] = float(feature[1])
        key_list = data_list[i][0].split(",")
        test_x_array[i][0] = int(key_list[0])
        test_x_array[i][1] = int(key_list[1])

    # os.makedirs("/tmp/model", exist_ok=True)

    print("is initialized:", initialized)
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            print("#############" + root + "/" + file)

    if not initialized:
        saver = tf.train.Saver()
        saver.restore(sess, model_dir + "/model.ckpt")
        initialized = True
        # saver.restore(sess, "C:/Users/yuchunfan/Desktop/ability_model" + "/model.ckpt")
    predict_ability = sess.run(y_value, feed_dict={x: test_x_array[:, 2:], keep_prob: 1.0})
    predict_list = sess.run(y, feed_dict={x: test_x_array[:, 2:], keep_prob: 1.0})
    # print("测试准确率：", list(zip(test_x_array[:, 0], test_x_array[:, 1], guess_value)))

    result_list = list(zip([int(x) for x in test_x_array[:, 0]],
                           [int(x) for x in test_x_array[:, 1]],
                           predict_ability,
                           (str(x) for x in predict_list)))
    result = list(map(lambda x: str({
        (x[0], x[1], x[2], ','.join(x[3][2:len(x[3]) - 1].split()))
    }), result_list))
    # sess.close()
    print(result)
    print('用时:', time.time() - last_moment)

    return initialized, True

# do_predict(False)
# do_predict(True)
