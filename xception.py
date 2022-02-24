from keras.applications.xception import Xception
from keras.datasets.cifar10 import load_data
from keras.layers import Input
#from keras.utils import to_categorical
#変更
from tensorflow.keras.utils import to_categorical
import numpy as np
import csv

def train():
    # CIFAR-10の画像データを取得（学習用とテスト用に分かれている）
    # x が画像，y がラベル
    (x_train, y_train), (x_test, y_test) = load_data()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # サンプルの x は256階調RGBのため0.0~1.0の範囲に変換
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.

    # 他クラス分類のために y を one-hot vector に変換
    n_labels = len(np.unique(y_train))
    y_train = to_categorical(y_train, n_labels)
    y_test = to_categorical(y_test, n_labels)
    print(y_train.shape, y_test.shape)

    # モデルを作成して学習を実行
    input = Input(shape=(32, 32, 3))
    model = Xception(weights=None, input_tensor=input, classes=n_labels)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))

    # 実行履歴（正解率の推移）をcsvで保存
    loss = h.history["loss"]
    val_loss = h.history["val_loss"]
    acc = h.history["acc"]
    val_acc = h.history["val_acc"]
    with open("history.csv", "wt", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["EPOCH", "ACC(TRAIN)", "ACC(TEST)", "LOSS(TRAIN)", "LOSS(TEST)"])
        for i in range(len(loss)):
            writer.writerow([i+1, acc[i], val_acc[i], loss[i], val_loss[i]])

    return model

if __name__ == "__main__":
  import sys, keras, tensorflow
  print("Python %s" % sys.version)
  print("Keras %s" % keras.__version__)
  print("TensorFlow %s" % tensorflow.__version__)

  model = train()
  model.save("cifer10+xception.hdf5")