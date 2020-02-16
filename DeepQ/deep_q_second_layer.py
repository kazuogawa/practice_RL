import numpy as np
from tensorflow.python import keras as K


def main():
    model = K.Sequential([
        # 活性化関数としてsigmoidを使っている
        K.layers.Dense(units=4, input_shape=(2,), activation="sigmoid"),
        K.layers.Dense(units=4)
    ])
    batch = np.random.rand(3,2)
    y = model.predict(batch)
    print(y.shape)


if __name__ == '__main__':
    main()
