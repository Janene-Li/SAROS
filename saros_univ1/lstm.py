from cgi import test
from venv import create
import numpy
import matplotlib.pyplot as plt
from requests import delete
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import  pandas as pd
import  os
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error
import csv


def create_dataset(dataset, look_back):
    #这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX),numpy.array(dataY)


def create_model():
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(None,1), dropout=0.2, return_sequences=True))    # 输出空间的维度（hidden size隐变量空间维度）； # 6可调
    model.add(LSTM(units=64, input_shape=(None,1), dropout=0.2, ))
    model.add(Dense(1))
    return model


def train_model(model):
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX_all, trainY_all, epochs=20, batch_size=20, verbose=2)  # epoch, batch_size=8,16,32,64 可调, batch size是可以同时输入16组数据
    model.save("univ1_trace/univ1_[{}-{}]_epoch-{}_fattree{}_x{}.h5".format(file_begin, file_end, test_epoch, fattree_k, x))    #os.path.join("DATA","Test" + ".h5")


def draw_pic(testY_all, testPredict, rmse):
    # print(testY_all.shape)
    plt.figure(figsize=(30, 5))  # figsize:确定画布大小 
    plt.plot(testY_all, c='blue', label='raw')
    plt.plot(testPredict, c='green', label='predict')

    plt.xlabel('epoch')
    plt.ylabel('hash')
    plt.legend(loc="upper right")  # 显示图例
    # 添加标题
    plt.title('switch{}, rmse={}'.format(switch_id, rmse))
    # 保存图形
    pic_dir = "univ1_trace/univ1_[{}-{}]_fattree{}_x{}/except{}/".format(file_begin, file_end, fattree_k, x, test_epoch)

    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    plt.savefig(pic_dir + "switch{}.jpg".format(switch_id))


def predict(model, scaler, testX_all, testY_all):
    # print(testY_all.shape)
    # trainPredict = model.predict(trainX_all)
    testPredict = model.predict(testX_all)

    #反归一化
    # trainPredict = scaler.inverse_transform(trainPredict)
    # trainY_all = scaler.inverse_transform(trainY_all)
    # testPredict = scaler.inverse_transform(testPredict)
    # testY_all = scaler.inverse_transform(testY_all)

    rmse = mean_squared_error(testY_all, testPredict)
    print("Test RMSE:", rmse)
    draw_pic(testY_all, testPredict, rmse)

def draw_switch_hash(x_axis_data, y1_axis_data, y2_axis_data, switch_id):

    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
    plt.figure(figsize=(40, 5))  # figsize:确定画布大小 
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x_axis_data, y1_axis_data, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='actual')
    plt.plot(x_axis_data, y2_axis_data, 'ro-', color='green', alpha=0.8, linewidth=1, label='predict')

    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('epoch')
    plt.ylabel('hash')

    pic_dir = save_data_dir + '[hash track]/'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    plt.savefig(pic_dir + 'switch{}.jpg'.format(switch_id))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
    
def evaluate_predict_hash(hash_file_dir):   
    logging.info("############################## evaluate predict p_threshold ##############################")

    """
    evaluation file to save
    """
    evaluation_file_name = save_data_dir + "[evaluate volume].csv"
    header = ["switch_id", "mse", "rsme", "mae", "mape", "smape"]

    if os.path.exists(evaluation_file_name):
        os.remove(evaluation_file_name)
    
    os.mknod(evaluation_file_name)
    logging.info("{} set already! ".format(evaluation_file_name))

    with open(evaluation_file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

    """
    read hash file
    """
    raw_df = pd.DataFrame(pd.read_csv(hash_file_dir))
    raw_df = raw_df[(raw_df.should_max_hash != -1)]

    """
    evaluate predict result
    """
    # calculate mse in all
    y_true = np.array(raw_df["should_max_hash"])
    y_pred = np.array(raw_df["sample_max_hash"])
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mape_ = mape(y_true, y_pred)
    smape_ = smape(y_true, y_pred)

    evaluation_data = [[-1, mse, rmse, mae, mape_, smape_]]
    with open(evaluation_file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(evaluation_data)
    
    # calculate mse in every switch
    grouped_df = raw_df.groupby('switch_id')

    for switch_df in grouped_df:

        switch_id = switch_df[0]
        switch_hash = switch_df[1]
        # switch_hash.sort_values(by="len_packets_through", inplace=True, ascending=True)

        logging.info("【switch {}】{} epoches ###################\n".format(switch_id, len(switch_hash)))

        # draw pic
        draw_switch_hash(switch_hash['epoch'], switch_hash['should_max_hash'], switch_hash['sample_max_hash'], switch_id)

        # calculate mse
        y_true = np.array(switch_hash["should_max_hash"])
        y_pred = np.array(switch_hash["sample_max_hash"])
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        mae = metrics.mean_absolute_error(y_true, y_pred)
        mape_ = mape(y_true, y_pred)
        smape_ = smape(y_true, y_pred)

        evaluation_data = [[switch_id, mse, rmse, mae, mape_, smape_]]
        with open(evaluation_file_name, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(evaluation_data)


if __name__ == '__main__':
    file_begin = 1
    file_end = 20
    test_epoch = 35
    fattree_k = 4

    x = 1000
    theta = 0.01
    alpha = 0.8

    len_core = int(pow(fattree_k / 2, 2))
    len_aggr = int(fattree_k / 2)
    len_edge = int(fattree_k / 2)
    switch_num = len_core + (len_aggr + len_edge) * fattree_k

    trainX_all = []
    trainY_all = []
    dataset_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    dataset_list.remove(test_epoch)
    for epoch in dataset_list:
        time_interval = epoch 
        print("time interval = {} #################".format(epoch))

        for switch_id in range(switch_num):

            dataframe = pd.read_csv("univ1_trace/univ1_[{}-{}]_epoch{}_fattree{}/x{}_should_max_hash/switch{}.csv".format(file_begin, file_end, time_interval, fattree_k, x, switch_id), usecols=[2]) # , usecols=[2], engine='python', skipfooter=3
            dataset = dataframe.values

            delete_index = []
            for i in range(len(dataset)):
                if dataset[i] == [-1.0]:
                    delete_index.append(i)
            dataset = np.delete(dataset, delete_index, axis=0)
            dataset = dataset.astype('float32')

            #归一化 
            scaler = MinMaxScaler(feature_range=(0, 1)) 
            dataset = scaler.fit_transform(dataset)

            train_size = int(len(dataset) * 1.0)
            trainlist = dataset[:train_size]
            # testlist = dataset[train_size:]

            look_back = 10  # 可调, 训练数据太少 look_back并不能过大

            trainX, trainY  = create_dataset(trainlist, look_back)
            # print(trainX.shape, trainY.shape)
            # testX, testY = create_dataset(testlist, look_back)

            trainX_all.append(trainX)
            trainY_all.append(trainY)
            # testX_all.append(testX)
            # testY_all.append(testY)
    trainX_all = np.concatenate(trainX_all,axis=0)
    trainY_all = np.concatenate(trainY_all,axis=0)

    trainX_all = numpy.reshape(trainX_all, (trainX_all.shape[0], trainX_all.shape[1], 1))   ## 样本数量， 长度， 数据维度


    # train
    model = create_model()
    train_model(model)

    testX_all = []
    testY_all = []

    for epoch in [test_epoch]:
        time_interval = epoch 
        print("time interval = {} #################".format(epoch))

        for switch_id in range(switch_num):

            dataframe = pd.read_csv("univ1_trace/univ1_[{}-{}]_epoch{}_fattree{}/x{}_should_max_hash/switch{}.csv".format(file_begin, file_end, time_interval, fattree_k, x, switch_id), usecols=[2]) # , usecols=[2], engine='python', skipfooter=3
            dataset = dataframe.values

            delete_index = []
            for i in range(len(dataset)):
                if dataset[i] == [-1.0]:
                    delete_index.append(i)
            dataset = np.delete(dataset, delete_index, axis=0)
            dataset = dataset.astype('float32')

            #归一化 
            scaler = MinMaxScaler(feature_range=(0, 1)) 
            dataset = scaler.fit_transform(dataset)

            testlist = dataset[0:]

            look_back = 10  # 可调, 训练数据太少 look_back并不能过大

            testX, testY = create_dataset(testlist, look_back)

            testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1] ,1 )) 

            # predict
            model = load_model("univ1_trace/univ1_[{}-{}]_epoch-{}_fattree{}_x{}.h5".format(file_begin, file_end, test_epoch, fattree_k, x))
            predict(model, scaler, testX, testY)
            # testX_all.append(testX)
            # testY_all.append(testY)

    # testX_all = np.concatenate(testX_all,axis=0)
    # testY_all = np.concatenate(testY_all,axis=0)

    # testX_all = numpy.reshape(testX_all, (testX_all.shape[0], testX_all.shape[1] ,1 )) 

    # # predict
    # model = load_model("univ1_trace/univ1_[{}-{}]_fattree{}_x{}.h5".format(file_begin, file_end, fattree_k, x))
    # predict(model, scaler, testX_all, testY_all)




 