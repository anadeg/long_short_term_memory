import numpy as np

from main import LstmParam, LstmNetwork


def make_window(sequence, size=-1):
    if size == -1:
        return sequence[:-1]
    else:
        return sequence[-size-1:-1]


def find_coefficient(x):
    max_value = np.max(x)
    coefficient = 1
    while coefficient < max_value:
        coefficient *= 10
    return coefficient


def example_0():
    np.random.seed(0)

    # x = np.array([[1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]])
    x = np.array([[(i+1) ** 2 for i in range(5)]])
    mem_cell_ct = 4
    x_dim = x.shape[-1] - 1

    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)

    # x_length = find_coefficient(x)
    # x = x / x_length
    x_train = np.array([make_window(x[-1])])
    y_train = np.array([x[-1][-1]])

    # lstm_net.fit_one(x_train, y_train, lr=0.1, error=10 ** 6)

    lstm_net.fit(x_train, y_train, lr=0.1, error=10**(-12))

    predictions = lstm_net.predict(x_train)
    print(predictions)

    # loss = np.inf
    # cur_iter = 0
    # while loss > 10 ** (-6) and cur_iter < 1000:
    #     cur_iter += 1
    #     print(f"iter {cur_iter}", end=": ")
    #     for ind in range(len(y_train)):
    #         lstm_net.insert_x(x_train[ind])
    #
    #     y_predicted = ", ".join(["% 2.5f" % (lstm_net.lstm_node_list[ind].state.h[0] * x_length) for ind in range(len(y_train))])
    #     print(f"y_predicted = [{y_predicted}]", end=", ")
    #
    #     loss = lstm_net.prediction_error(y_train, Error)
    #     print("loss:", "%.3e" % loss)
    #     lstm_param.update(lr=0.1)
    #     lstm_net.x_clear()
    #
    # result = ", ".join(["% 2.5f" % (lstm_net.lstm_node_list[ind].state.h[0]) for ind in range(len(y_train))])
    # print(f'final prediction: [{result}]')
    #
    # lstm_net.predict_one()


if __name__ == "__main__":
    example_0()
