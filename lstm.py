import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


TRAIN_SIZE = 50
SPLIT_SIZE = 5
SEQUENCE_AMOUNT = 4


def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)


def predict_sequence(x_predict, amount=3):
    result = []
    for i in range(amount):
        yhat = model.predict(x_predict, verbose=0)
        x, y, z = x_predict.shape
        x_predict = np.append(x_predict, yhat)
        x_predict = x_predict[1:]
        x_predict = x_predict.reshape(x, y, z)
        result.append(yhat[0][0])
    return result


def function(x, type='linear'):
    match type:
        case 'linear':
            return 0.5 * x - 10
        case 'geometrical':
            return 1.5 ** x
        case 'square':
            return x ** 2
        case 'period':
            return x % 5
        case _:
            return x


if __name__ == '__main__':
    TYPE = 'geometrical'
    raw_seq = [function(i + 1, type=TYPE) for i in range(TRAIN_SIZE)]
    n_steps = SPLIT_SIZE
    X, y = split_sequence(raw_seq, n_steps)

    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=200, verbose=0)

    start = TRAIN_SIZE - SPLIT_SIZE
    end = TRAIN_SIZE
    x_input = array([function(i + 1, type=TYPE) for i in range(start, end)])
    x_input = x_input.reshape((1, n_steps, n_features))

    # x_output = array([10 * (i + 1) for i in range(TRAIN_SIZE, TRAIN_SIZE + SEQUENCE_AMOUNT)])
    true_values = ', '.join([str(function(i + 1, type=TYPE)) for i in range(TRAIN_SIZE, TRAIN_SIZE + SEQUENCE_AMOUNT)])

    print(f'true values:\t[{true_values}]')
    prediction = predict_sequence(x_input, amount=SEQUENCE_AMOUNT)
    prediction_values = ', '.join([str(pred) for pred in prediction])
    print(f'prediction:\t\t[{prediction_values}]')