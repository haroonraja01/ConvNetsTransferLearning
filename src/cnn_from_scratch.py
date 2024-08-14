import numpy as np


def relu(x):
    x[x <= 0] = 0
    return x


class CNN_numpy:
    def __init__(self, k_dim_1=None, k_dim_2=None):
        if k_dim_1 is None and k_dim_2 is None:
            self.k_dim_1 = 5  # kernel dimension
            self.k_dim_2 = 5  # kernel dimension
        else:
            self.k_dim_1 = k_dim_1
            self.k_dim_2 = k_dim_2
        self.kernel = np.random.randn(self.k_dim_1, self.k_dim_2)

    def conv2d(self, x):
        d1 = np.shape(x)[0]
        d2 = np.shape(x)[1]
        y = np.zeros((d1 - self.k_dim_1 + 1, d2 - self.k_dim_2 + 1))
        for i in range(d1 - self.k_dim_1 + 1):
            for j in range(d2 - self.k_dim_2 + 1):
                # y[i][j] = np.sum(x[i:i + self.k_dim_1, j:j + self.k_dim_2] * self.kernel)
                y[i, j] = np.sum(x[i:i+self.k_dim_1, j:j+self.k_dim_2] * self.kernel)
        return y


if __name__ == '__main__':
    cnn_obj = CNN_numpy(k_dim_1=5, k_dim_2=5)
    x_ip = np.random.randn(11, 11)
    output = cnn_obj.conv2d(x=x_ip)
    output = relu(output)
    print(output)

