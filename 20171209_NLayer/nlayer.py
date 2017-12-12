# -*- coding: utf-8 -*-
"""
deeplerning に必要なデータを準備する
"""
from typing import List, Dict, Tuple
import os
import sys
import os.path
import pickle
import time
from scipy import ndimage
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

LARGE_FILE_PATH = '../notMNIST_large/'
SMALL_FILE_PATH = '../notMNIST_small/'
ALPHABETS_LIST: Dict[str, int] = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
                                  'I': 8, 'J': 9}
NUM_KIND_LABEL = 10
PIXEL_DEPTH = 256.0
IMG_SIZE = 28


def list_files(root_dir) -> List[List[str]]:
    """
    画像ファイルのリストを作る
    """
    directories: List[str] = os.listdir(root_dir)
    result: List[List[str]] = [None] * 10
    for directory in directories:
        char_num: int = ALPHABETS_LIST[directory]
        one: List[str] = []
        directory = os.path.join(root_dir, directory)
        if not os.path.isdir(directory):
            continue
        file_names = os.listdir(directory)
        for file_name in file_names:
            one.append(os.path.join(directory, file_name))
        result[char_num] = one
    return result

def load_letter(path) -> np.matrix:
    """
    画像を行列にする
    """
    # 28x28で、0-255の行列になる
    matrix = ndimage.imread(path)
    # float型に変換
    matrix = matrix.astype(float)
    # -1 ~ 1 に変換
    matrix = (matrix - PIXEL_DEPTH / 2) / (PIXEL_DEPTH / 2)
    return matrix

class LetterData:
    """
    文字データ
    """
    letter: np.matrix
    label: np.matrix

class Datasets:
    """
    データセット
    """
    train: LetterData
    valid: LetterData
    test: LetterData

def load_data(path_lists: List[List[str]]) -> LetterData:
    """
    データセットをロードする
    """
    num_kind = len(path_lists)
    num = 0
    for i, _ in enumerate(path_lists):
        num += len(path_lists[i])
    letter_list = np.ndarray(shape=(num, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    label_list = np.zeros(shape=(num, num_kind), dtype=np.float32)
    num = 0
    for i, _ in enumerate(path_lists):
        for path_list in path_lists[i]:
            try:
                # 文字の行列
                letter = load_letter(path_list)
                letter_list[num, :, :] = letter
                # ラベル
                label_list[num, i] += 1.0
                num += 1
            except IOError as exp:
                print('Could not read:', path_list,
                      ':', exp, '- it\'s ok, skipping.')
    letter_list = letter_list.reshape((-1, IMG_SIZE * IMG_SIZE)).astype(np.float32)
    result = LetterData()
    result.letter = letter_list
    result.label = label_list
    return result

def prepare_datasets(train_size: int, valid_size: int, test_size: int,
                    paths: List[List[str]]) -> Datasets:
    """
    データセットの抽出
    """

    train_files: List[List[str]] = [None] * 10
    valid_files: List[List[str]] = [None] * 10
    test_files: List[List[str]] = [None] * 10
    train_index = int(train_size / 10)
    valid_index = int(valid_size / 10) + train_index
    test_index = int(test_size / 10) + valid_index
    for i in range(10):
        train_files[i] = paths[i][0:train_index]
        valid_files[i] = paths[i][train_index:valid_index]
        test_files[i] = paths[i][valid_index:test_index]
    result = Datasets()
    result.train = load_data(train_files)
    result.valid = load_data(valid_files)
    result.test = load_data(test_files)
    return result

# def accuracy(predictions, labels):
#     """
#     正答率を示す
#     """
#     return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def learningNormal(datasets: Datasets, train_subset: int):
    """
    単純な学習
    """
    graph = tf.Graph()

    # Tensor Flow の準備
    with graph.as_default():
        # データセットの定数
        dataset = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE])
        label = tf.placeholder(tf.float32, [None, NUM_KIND_LABEL])

        # ウェイトは最初は乱数値
        # INPUT x OUTPUT の行列
        weight_n = tf.Variable(tf.truncated_normal([IMG_SIZE * IMG_SIZE, NUM_KIND_LABEL]))
        bias_n = tf.Variable(tf.zeros([NUM_KIND_LABEL]))
        hidden_n = tf.nn.relu(tf.matmul(dataset, weight_n) + bias_n)

        for i in range(int(sys.argv[1])):
            weight_np = tf.Variable(tf.truncated_normal([NUM_KIND_LABEL, NUM_KIND_LABEL]))
            bias_np = tf.Variable(tf.zeros([NUM_KIND_LABEL]))
            hidden_np = tf.nn.relu(tf.matmul(hidden_n, weight_np) + bias_np)
            hidden_n = hidden_np

        # 結果
        weight0 = tf.Variable(tf.zeros([NUM_KIND_LABEL, NUM_KIND_LABEL]))
        bias0 = tf.Variable(tf.zeros([NUM_KIND_LABEL]))
        logit = tf.nn.softmax(tf.matmul(hidden_n, weight0) + bias0)

        # 損失の演算
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,
        #                                                               logits=logits))
        loss = -tf.reduce_sum(label * tf.log(logit))

        # 最適化
        # 微分が最小の損失となる物を見つける
        train_step = tf.train.AdamOptimizer().minimize(loss)

        # トレーニングの結果
        correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_steps = 6000

    # 実行
    print("step,loss,traning accuracy,validation accuracy,test accuracy")
    with tf.Session(graph=graph) as session:
        # これは一度の演算で、パラメータは、乱数の重みを持つ行列と、ゼロのバイアスで初期化される。
        tf.global_variables_initializer().run()

        starttime = time.time()

        for step in range(num_steps):

            c_loss, _, c_accurancy = session.run([loss, train_step, accuracy],
                                                  feed_dict={dataset: datasets.train.letter[:train_subset],
                                                             label: datasets.train.label[:train_subset]})
            # _, c_loss, predictions = session.run([optimizer, loss, train_prediction])

            if step % 100 == 0:
                laptime = time.time()
                interval = laptime - starttime
                valid_occuracy = session.run([accuracy],
                                              feed_dict={dataset: datasets.valid.letter,
                                                         label: datasets.valid.label})
                # train_accuracy = accuracy(predictions, datasets.train.label[:train_subset, :])
                # valid_accuracy = accuracy(valid_prediction.eval(), datasets.valid.label)
                # test_accuracy = accuracy(test_prediction.eval(), datasets.test.label)
                print("%10.3f,%5d,%10.6f,%.6f,%.6f,%.6f" %
                      (interval, step, c_loss, c_accurancy, valid_occuracy[0], 0))

def main():
    # データセットの準備
    # paths = list_files(LARGE_FILE_PATH)
    # train_size = 100000
    # valid_size = 1000
    # test_size = 1000
    # datasets = prepare_datasets(train_size, valid_size, test_size, paths)
    # with open("dataset.pickle", 'wb') as pickle_file:
    #     pickle.dump(datasets, pickle_file, pickle.HIGHEST_PROTOCOL)
    #     pickle_file.close()

    with open("dataset.pickle", 'rb') as pickle_file:
        datasets: Datasets = pickle.load(pickle_file)
        pickle_file.close()

    learningNormal(datasets, 100000)



main()
