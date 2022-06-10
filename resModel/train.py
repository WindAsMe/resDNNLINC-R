from tensorflow.keras import callbacks
import datetime
import joblib
from os import path
from util import helps
from sklearn.preprocessing import MinMaxScaler
from resModel.resNet import ResNet50Regression
from bench import benchmark


def training(model, X, y):
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # compute running time
    starttime = datetime.datetime.now()

    model.fit(X, y, epochs=100, batch_size=5000, verbose=2, callbacks=[callbacks.EarlyStopping(monitor='val_loss',
                                                                                              patience=10, verbose=2,
                                                                                              mode='auto')],
              validation_split=0.1)
    endtime = datetime.datetime.now()
    print("train time: ", endtime - starttime)
    return model


def Model_Build(Dim, func_num, bias, dense_size, this_path, sample_size=1):
    F = benchmark.Function(func_num)
    func = F.get_func()
    scale_range = F.get_info()
    for i in range(sample_size):
        samples = helps.sampling(Dim, scale_range, bias, sample_size)
        path_samples = this_path + '/data/sample/f' + str(func_num)
        helps.write(samples, path_samples)

        neighbors, dense_scale = helps.dense_sampling(samples[0], bias, dense_size)
        train_X, train_y = helps.data_generate(neighbors, func)
        path_scaler_x = this_path + '/data/model/f' + str(func_num) + '/scaleX_' + str(i)
        path_scaler_y = this_path + '/data/model/f' + str(func_num) + '/scaley_' + str(i)
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_x.fit(train_X)
        scaler_y.fit(train_y)
        train_X_scale = scaler_x.transform(train_X)
        train_y_scale = scaler_y.transform(train_y)

        joblib.dump(scaler_x, path_scaler_x)
        joblib.dump(scaler_y, path_scaler_y)
        '''
        model config and train
        '''
        path_model = this_path + '/data/model/f' + str(func_num) + '/resNet_' + str(i) + '.h5'
        model = training(ResNet50Regression(Dim), train_X_scale, train_y_scale)
        model.save(path_model)
