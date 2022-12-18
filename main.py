import numpy as np


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_derivative(values):
    return values * (1 - values)


def tanh_derivative(values):
    return 1. - values ** 2


def rand_arr(*args):
    np.random.seed(0)
    return np.random.rand(*args)


class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct

        self.wg = rand_arr(mem_cell_ct, concat_len)
        self.wi = rand_arr(mem_cell_ct, concat_len)
        self.wf = rand_arr(mem_cell_ct, concat_len)
        self.wo = rand_arr(mem_cell_ct, concat_len)

        self.bg = rand_arr(mem_cell_ct)
        self.bi = rand_arr(mem_cell_ct)
        self.bf = rand_arr(mem_cell_ct)
        self.bo = rand_arr(mem_cell_ct)

        self.wg_diff = np.zeros((mem_cell_ct, concat_len))
        self.wi_diff = np.zeros((mem_cell_ct, concat_len))
        self.wf_diff = np.zeros((mem_cell_ct, concat_len))
        self.wo_diff = np.zeros((mem_cell_ct, concat_len))
        self.bg_diff = np.zeros(mem_cell_ct)
        self.bi_diff = np.zeros(mem_cell_ct)
        self.bf_diff = np.zeros(mem_cell_ct)
        self.bo_diff = np.zeros(mem_cell_ct)

    def update(self, lr=1):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff


class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)


class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        self.state = lstm_state
        self.param = lstm_param
        self.x_concatenated = None

    def bottom_data_is(self, x, s_prev=None, h_prev=None):
        if s_prev is None:
            s_prev = np.zeros_like(self.state.s)
        if h_prev is None:
            h_prev = np.zeros_like(self.state.h)

        self.s_prev = s_prev
        self.h_prev = h_prev

        xc = np.hstack((x, h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = np.tanh(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o

        self.x_concatenated = xc

    def top_diff(self, top_diff_h, top_diff_s):
        ds = self.state.o * top_diff_h + top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        di_input = sigmoid_derivative(self.state.i) * di
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = tanh_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg

        self.param.wi_diff += np.outer(di_input, self.x_concatenated)
        self.param.wf_diff += np.outer(df_input, self.x_concatenated)
        self.param.wo_diff += np.outer(do_input, self.x_concatenated)
        self.param.wg_diff += np.outer(dg_input, self.x_concatenated)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        dxc = np.zeros_like(self.x_concatenated)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:]


class LstmNetwork():
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []

        self.x_list = []

    def prediction_error(self, y_list, loss_layer):
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1

        loss = loss_layer.error(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.error_diff(self.lstm_node_list[idx].state.h, y_list[idx])

        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff(diff_h, diff_s)
        idx -= 1

        while idx >= 0:
            loss += loss_layer.error(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h = loss_layer.error_diff(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
            self.lstm_node_list[idx].top_diff(diff_h, diff_s)
            idx -= 1

        return loss

    def x_clear(self):
        self.x_list = []

    def insert_x(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

        idx = len(self.x_list) - 1
        if idx == 0:
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)

    def feature_engineering(self, x, y):
        self.coefficient = np.maximum(self.find_coefficient(x), self.find_coefficient(y))
        x = x / self.coefficient
        y = y / self.coefficient
        return x, y

    def fit(self, x_train, y_train, lr=0.01, error=10 ** (-6)):
        x_train, y_train = self.feature_engineering(x_train, y_train)

        loss = np.inf
        cur_iter = 0
        while loss > error:
            cur_iter += 1
            for ind in range(len(y_train)):
                self.insert_x(x_train[ind])

            y_predicted = ", ".join(
                ["% 2.5f" % (self.lstm_node_list[ind].state.h[0] * self.coefficient) for ind in range(len(y_train))])
            print(f"y_predicted = [{y_predicted}]", end=", ")

            loss = self.prediction_error(y_train, Error)
            print("loss:", "%.3e" % loss)
            self.lstm_param.update(lr=lr)
            self.x_clear()

    # def predict_one(self):
    #     result = ", ".join(["% 2.5f" % (self.lstm_node_list[ind].state.h[0] * self.coefficient)
    #                         for ind in range(len(self.y_train))])
    #     print(f'final prediction: [{result}]')

    def predict(self, x_train, amount=3):
        x_train, _ = self.feature_engineering(x_train, [-np.inf])

        prediction = []

        for i in range(amount):
            if i != 0:
                for ind in range(len(x_train)):
                    self.insert_x(x_train[ind])
            to_slide = np.array([self.lstm_node_list[ind].state.h[0]
                            for ind in range(len(x_train))]).reshape(-1, 1)
            x_train = x_train[:, 1:]
            x_train = np.c_[x_train, to_slide]
            y_train = to_slide
            prediction.append(to_slide * self.coefficient)
            self.x_clear()
        return prediction

    @staticmethod
    def find_coefficient(x):
        max_value = np.max(x)
        coefficient = 1
        while coefficient < max_value:
            coefficient *= 10
        return coefficient

    @staticmethod
    def make_window(sequence, size=-1):
        if size == -1:
            return sequence[:-1]
        else:
            return sequence[-size - 1:-1]


class Error:
    @classmethod
    def error(self, predicted, label):
        return (predicted[0] - label) ** 2

    @classmethod
    def error_diff(self, predicted, label):
        diff = np.zeros_like(predicted)
        diff[0] = 2 * (predicted[0] - label)
        return diff
