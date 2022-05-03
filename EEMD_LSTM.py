import numpy as np
import pandas as pd

import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest
from keras.callbacks import ReduceLROnPlateau
from pandas import concat
from PyEMD import EEMD
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Flatten, SimpleRNN, GRU
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers import Activation
from keras import initializers
from scipy import interpolate
import matplotlib.pyplot as plt
import math
from keras import Input, Model
from keras.layers import Dense
from keras.models import load_model


def create_dataset(dataset, look_back=10):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


def visualize(history):
    plt.rcParams['figure.figsize'] = (10.0, 6.0)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def LSTM_Model(trainX, trainY, i):
    # filepath = '../lbw5/' + str(i) + '-{epoch:02d}-{val_acc:.2f}.h5'
    # checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1,save_best_only=False,mode='auto',period=10)
    # callbacks_list = [checkpoint]
    model = Sequential()
    model.add(LSTM(64, input_shape=(trainX.shape[1], trainX.shape[2])))  # 已经确定10步长
    '''
    ,return_sequences = True
    如果设置return_sequences = True，该LSTM层会返回每一个time step的h，
    那么该层返回的就是1个由多个h组成的2维数组了，如果下一层不是可以接收2维数组
    的层，就会报错。所以一般LSTM层后面接LSTM层的话，设置return_sequences = True，
    如果接全连接层的话，设置return_sequences = False。
    '''
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=500, batch_size=64, validation_split=0.1, verbose=2, shuffle=True)
    return (model)

def GRU_Model(trainX, trainY, i):
    # filepath = '../lbw5/' + str(i) + '-{epoch:02d}-{val_acc:.2f}.h5'
    # checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1,save_best_only=False,mode='auto',period=10)
    # callbacks_list = [checkpoint]
    model = Sequential()
    model.add(GRU(96, input_shape=(trainX.shape[1], trainX.shape[2])))  # 已经确定10步长
    # model.add(Dense(10, activation='elu'))
    model.add(Dense(10, activation='linear'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=500, batch_size=64, validation_split=0.1, verbose=2, shuffle=True)
    return (model)

def RNN_Model(trainX, trainY, i):
    # filepath = '../lbw5/' + str(i) + '-{epoch:02d}-{val_acc:.2f}.h5'
    # checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1,save_best_only=False,mode='auto',period=10)
    # callbacks_list = [checkpoint]
    model = Sequential()
    model.add(SimpleRNN(32, input_shape=(trainX.shape[1], trainX.shape[2])))  # 已经确定10步长
    '''
    ,return_sequences = True
    如果设置return_sequences = True，该LSTM层会返回每一个time step的h，
    那么该层返回的就是1个由多个h组成的2维数组了，如果下一层不是可以接收2维数组
    的层，就会报错。所以一般LSTM层后面接LSTM层的话，设置return_sequences = True，
    如果接全连接层的话，设置return_sequences = False。
    '''
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=1000, batch_size=64, validation_split=0.1, verbose=2, shuffle=True)
    return (model)


def DNN_Model(trainX, trainY, i):
    # filepath = '../lbw5/' + str(i) + '-{epoch:02d}-{val_acc:.2f}.h5'
    # checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1,save_best_only=False,mode='auto',period=10)
    # callbacks_list = [checkpoint]
    init = initializers.glorot_uniform(seed=1)
    model = Sequential()
    model.add(Dense(64, input_shape=(trainX.shape[1], trainX.shape[2]), activation='relu', kernel_initializer=init))
    model.add(Flatten())
    model.add(Dense(10, activation='linear', kernel_initializer=init))
    model.add(Dense(1, activation='linear', kernel_initializer=init))
    model.compile(loss='mse', optimizer='adam')
    # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
    #                                   epsilon=0.0001, cooldown=0, min_lr=0)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, mode='auto')
    model.fit(trainX, trainY, epochs=500, batch_size=64, validation_split=0.1, verbose=2, shuffle=True)  # callbacks=[reduce_lr]
    return (model)


def BP_Model(trainX, trainY, i):
    # filepath = '../lbw5/' + str(i) + '-{epoch:02d}-{val_acc:.2f}.h5'
    # checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1,save_best_only=False,mode='auto',period=10)
    # callbacks_list = [checkpoint]
    init = initializers.glorot_uniform(seed=1)
    model = Sequential()
    model.add(Dense(64, input_shape=(trainX.shape[1], trainX.shape[2]), activation='relu', kernel_initializer=init))
    model.add(Flatten())
    model.add(Dense(1, activation='linear', kernel_initializer=init))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=500, batch_size=64, validation_split=0.1, verbose=2, shuffle=True)
    return (model)


def plot_curve(true_data, predicted):
    # rmse=format(RMSE(test,prediction),'.4f')
    # mape=format(MAPE(test,prediction),'.4f')
    plt.plot(true_data, label='True data')
    plt.plot(predicted, label='Predicted data')
    plt.legend()
    # plt.text(1, 1, 'RMSE:' + str(rmse)+' \n '+'MAPE:'+str(mape), color = "r",style='italic', wrap=True)
    # plt.text(2, 2, "RMSE:" + str(format(RMSE(true_data,predicted),'.4f'))+" \n "+"MAPE:"+str(format(MAPE(true_data,predicted),'.4f')), style='italic', ha='center', wrap=True)
    # plt.savefig('result_EEMD_LSTM_E5B16.png')
    plt.show()


def RMSE(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return rmse


def MAE(test, predicted):
    dif = mean_absolute_error(test, predicted)
    return dif

def MAPE(Y_true, Y_pred):
    Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
    return np.mean(np.fabs((Y_true - Y_pred) / Y_true)) * 100


if __name__ == '__main__':

    # plt.rcParams['figure.figsize'] = (10.0, 5.0)  # set default size of plots
    # plt.rcParams['image.interpolation'] = 'nearest'
    # plt.rcParams['image.cmap'] = 'gray'
    # ##########################读取数据###########################
    df = pd.read_csv('C:/Users/Lenovo/Desktop/pychram_program/LSTM/1703_4_KMFT.txt')
    data = df.values
    dataset = 840.4-data
    # dataset = data
    # ###########################归一化###########################
    scaler_data = MinMaxScaler(feature_range=(0, 1))
    Wear_D = scaler_data.fit_transform(dataset)
    # ###########################EEMD分解###########################
    eemd = EEMD()
    eemd.noise_seed(12345)
    eemd.eemd(Wear_D.reshape(-1), None, 5)
    imfs, res = eemd.get_imfs_and_residue()
    res.reshape(1, len(res))
    imfs = np.concatenate((imfs, res.reshape(1, len(res))))
    # ###########################EEMD分解结果可视化###########################
    # i = 1
    # for imf in imfs:
    #     plt.subplot(len(imfs), 1, i)
    #     plt.plot(imf)
    #     i += 1
    # plt.savefig('result_imf.png')
    # plt.show()
    # ##########################模型预测###########################
    # print(imfs.shape)
    c = int(len(Wear_D) * 0.8)
    lookback_window = 20
    imfs_prediction = []
    test = np.zeros([len(dataset) - c - lookback_window, 1])
    i = 1
    for imf in imfs:
        print('-' * 45)
        print('This is  ' + str(i) + '  time(s)')
        print('*' * 45)

        # 分割2/3数据作为测试
        temp = imf.reshape(len(imf), 1)
        train_size = c
        test_size = len(temp) - train_size
        train, test = temp[0:train_size, :], temp[train_size:len(temp), :]
        # 预测数据步长为1,一个预测一个，1->1
        look_back = lookback_window
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        # 重构输入数据格式 [samples, time steps, features] = [93,1,1]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        model = LSTM_Model(trainX, trainY, i)
        # model = load_model('C:/Users/Lenovo/Desktop/pychram_program/LSTM/LSTM3.96_model.h5')
        # model.save('C:/Users/Lenovo/Desktop/pychram_program/LSTM/modsave/EEMD-LSTM-imf' + str(i) + '-100.h5')
        prediction_Y = model.predict(testX)
        imfs_prediction.append(prediction_Y)
        i += 1;
    imfs_prediction = np.array(imfs_prediction)

    imfs_predictionplot = np.empty_like(imfs_prediction)
    # imfs_predictionplot[:, :, :] = np.nan
    # imfs_predictionplot[:, train_size + (lookback_window * 1) + 1:imf.shape[0] - 1, :] = imfs_prediction
    # #############################坐标整合############################
    ipp = []
    for i in range(imfs_prediction.shape[0]):
        # imfs_predictionplot = np.empty_like([imf.shape[0], 1])
        p = imf.reshape(imf.shape[0], 1)
        imfs_predictionplot = np.empty_like(p)
        imfs_predictionplot[:, :] = np.nan
        imfs_predictionplot[train_size + (lookback_window * 1):imf.shape[0]-1, :] = imfs_prediction[i]
        ipp.append(imfs_predictionplot)

    # ############################每个分量的预测结果可视化#####################
    # i = 1
    # for imf in imfs:
    #     plt.subplot(len(imfs), 1, i)
    #     plt.plot(imf)
    #     plt.plot(ipp[i-1])
    #     i += 1
    # plt.savefig('result_imf.png')
    # plt.show()
    # ###########################最终预测结果的可视化###########################
    predict_plot = np.zeros([ipp[0].shape[0], 1])
    for i in range(imfs.shape[0]):
        predict_plot = predict_plot + ipp[i]
    pre_plot = scaler_data.inverse_transform(predict_plot)
    plt.plot(pre_plot)
    plt.plot(dataset)
    plt.show()
    # ##########################评价指标计算###########################
    start = train_size + (lookback_window * 1)
    end = imf.shape[0] - 1
    # plot_curve(test, prediction)
    print('MAE: ', MAE(dataset[start:end], pre_plot[start:end])*100)
    print('MAPE: ', MAPE(dataset[start:end], pre_plot[start:end]))
    print('RMSE: ', RMSE(dataset[start:end], pre_plot[start:end])*100)
# ###########################预测结果保存###########################
#     f = open('predict_EEMD-LSTM_KMFT_median.txt', 'wb')
#     for i in range(len(pre_plot)):
#         f.write(str.encode(str(pre_plot[i]) + '\n'))
#     f.close()
# ###########################EEMD分解结果保存###########################
#     i = 1
#     for imf in imfs:
#         f = open('C:/Users/Lenovo/Desktop/pychram_program/LSTM/imfs_results/LSTMimfs-'+str(i)+'.txt', 'wb')
#         for j in range(len(imf)):
#             f.write(str.encode(str(imf[j]) + '\n'))
#         f.close()
#         i += 1
# # ###########################各分量预测结果保存###########################
#     num = imfs_prediction.shape[0]
#     length = imfs_prediction.shape[1]
#     imfpred = imfs_prediction.reshape(num, length)
#     for i in range(num):
#         f = open('C:/Users/Lenovo/Desktop/pychram_program/LSTM/imfs_results/LSTMimfs_predict-'+str(i+1)+'.txt', 'wb')
#         for j in range(len(imf)):
#             f.write(str.encode(str(imf[j]) + '\n'))
#         f.close()