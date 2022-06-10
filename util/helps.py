import random
import numpy as np


def sampling(Dim, scale_range, bias, size):
    samples = []
    for n in range(size):
        sample = []
        for i in range(Dim):
            sample.append(random.uniform(scale_range[0] + bias, scale_range[1] - bias))
        samples.append(sample)
    return np.array(samples)


def dense_sampling(base_point, bias, size):
    scale = []
    for coordinate in base_point:
        ub = coordinate + bias
        lb = coordinate - bias
        scale.append([lb, ub])
    neighbors = []
    for n in range(size):
        neighbor = []
        for i in range(len(base_point)):
            neighbor.append(random.uniform(scale[i][0], scale[i][1]))
        neighbors.append(neighbor)
    return np.array(neighbors), scale


def data_generate(X, func):
    y = []
    for x in X:
        y.append(func(x))
    return np.array(X), np.array(y).reshape(-1, 1)


def write(data, path):
    with open(path, 'a+') as file:
        for sample in data:
            file.write('[')
            for e in sample:
                file.write(str(e) + ', ')
            file.write(']')
            file.write('\n')
        file.close()


