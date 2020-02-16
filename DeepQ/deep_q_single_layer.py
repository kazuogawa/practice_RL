import numpy as np
from tensorflow.python import keras as K


def main():
    # K.Sequentialは複数の層をまとめるためのモジュール
    # 今回は1層のニューラルネットワーク
    model = K.Sequential([
        # 入力に対し重みをかけ、バイアスを足す処理を行う
        # 入力サイズは座標と同じく2,出力サイズは行動価値と同じ4
        K.layers.Dense(units=4, input_shape=(2,))
    ])
    weight, bias = model.layers[0].get_weights()
    print("Weight shape is {}.".format(weight.shape))
    print("Bias shape is {}.".format(bias.shape))
    x = np.random.rand(1, 2)
    y = model.predict(x)
    print("x is ({}) and y is ({})".format(x.shape, y.shape))


if __name__ == '__main__':
    main()
