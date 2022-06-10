import copy
import numpy as np
import joblib
from keras.models import load_model


def LIEM_model(Dim, path, func_num, samples, delta, epsilon, max_len):
    check_times = len(samples)
    adjacent_matrix = np.zeros((Dim, Dim))
    scaler_Xs, scaler_ys, models = Load(path, func_num, len(samples))
    for i in range(Dim):
        for j in range(i+1, Dim):
            for check in range(check_times):
                adjacent_matrix[i][j] = max(adjacent_matrix[i][j], Differential(samples[check], i, j, delta,
                                                                                models[check], scaler_Xs[check]))
                adjacent_matrix[j][i] = adjacent_matrix[i][j]

    groups = DECC_DG(Dim, adjacent_matrix, epsilon, max_len)
    return groups


def Load(path, func_num, sample_IDs):
    models = []
    scaler_Xs = []
    scaler_ys = []
    for i in range(sample_IDs):
        model_path = path + '/data/model/f' + str(func_num) + '/resNet_' + str(i) + '.h5'
        scaleX_path = path + '/data/model/f' + str(func_num) + '/scaleX_' + str(i)
        scaley_path = path + '/data/model/f' + str(func_num) + '/scaley_' + str(i)
        scaler_X = joblib.load(scaleX_path)
        scaler_y = joblib.load(scaley_path)
        model = load_model(model_path)
        scaler_Xs.append(scaler_X)
        scaler_ys.append(scaler_y)
        models.append(model)
    return scaler_Xs, scaler_ys, models


def DECC_DG(Dim, adj_matrix, epsilon, max_len):
    groups = CCDE(Dim)
    for i in range(len(groups)-1):
        for j in range(i+1, len(groups)):
            if i < len(groups)-1 and j < len(groups) and len(groups[i]) < max_len and adj_matrix[i][j] > epsilon:
                groups[i].extend(groups.pop(j))
                j -= 1
    return groups


def Differential(sample, e1, e2, delta, model, scale_X):
    modify_sample = scale_X.transform(sample)
    a = model.predict([modify_sample])[0]

    index_b = copy.deepcopy(sample)
    index_b[e1] += delta
    modify_index_b = scale_X.transform(index_b)
    b = model.predict([modify_index_b])[0]

    index_c = copy.deepcopy(sample)
    index_c[e2] += delta
    modify_index_c = scale_X.transform(index_c)
    c = model.predict([modify_index_c])[0]

    index_c[e1] += delta
    modify_index_d = scale_X.transform(index_c)
    d = model.predict([modify_index_d])[0]

    return np.abs((d + a) - (c + b))


def CCDE(N):
    groups = []
    for i in range(N):
        groups.append([i])
    return groups
