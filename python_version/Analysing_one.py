import pandas as pd
import sys
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from tensorflow.keras.models import Sequential, load_model
import csv


def evaluate_HH(groundtruth_dir, simulating_dir):
    """
    # simulating result -> group
    """
    # effective_x,
    # epoch,
    # global_threshold,
    # id,
    # sampling_rate

    Simulating_Heavy_Hitter_Set = pd.read_csv(simulating_dir)
    Simulating_Heavy_Hitter_Set = Simulating_Heavy_Hitter_Set.loc[:, ~Simulating_Heavy_Hitter_Set.columns.str.contains('Unnamed')]
    Simulating_Heavy_Hitter_Set.info()
    simulate_epoch_group = Simulating_Heavy_Hitter_Set.groupby('epoch')

    """
    # groundtruth -> group
    """
    # epoch,
    # global_T,
    # id,
    # packets_in_epoch,
    # packets_in_flow,
    # theta

    Groundtruth_Heavy_Hitter_Set = pd.read_csv(groundtruth_dir)
    Groundtruth_Heavy_Hitter_Set = Groundtruth_Heavy_Hitter_Set.loc[:, ~Groundtruth_Heavy_Hitter_Set.columns.str.contains('Unnamed')]
    Groundtruth_Heavy_Hitter_Set.info()
    groundtruth_epoch_group = Groundtruth_Heavy_Hitter_Set.groupby('epoch')

    """
    check HH
    """
    # epoch in all
    epoch_count = Groundtruth_Heavy_Hitter_Set['epoch'].max()
    logging.info("epoch count is {}".format(epoch_count))

    # result to save
    result_df = pd.DataFrame()

    # check every epoch
    if epoch_count != 0:

        for epoch_i in range(0, epoch_count):
            # if epoch_i == 5:
            #     break
            P = 0
            N = 0
            TP = 0
            TN = 0
            FP = 0
            FN = 0

            len_groundtruth = len(Groundtruth_Heavy_Hitter_Set.query("epoch=={}".format(epoch_i)))
            len_simulate = len(Simulating_Heavy_Hitter_Set.query("epoch=={}".format(epoch_i)))
            logging.info("############################## epoch = {}; len_groundtruth = {}; len_simulate = {} ##############################".format(epoch_i, len_groundtruth, len_simulate))
            
            N_plus_P = pd.DataFrame(groundtruth_epoch_group.get_group(epoch_i)['packets_in_epoch']).iloc[0].at['packets_in_epoch']
            logging.info("N_plus_P = {}".format(N_plus_P))
            
            if len_groundtruth == 0 and len_simulate == 0:
                continue

            elif len_groundtruth == 0 and len_simulate != 0:
                P = 0
                N = N_plus_P - P
                TP = 0  # 被模型预测为正类的正样本
                FP = len_simulate  # 被模型预测为正类的负样本
                FN = 0  # 被模型预测为负类的正样本
                TN = N - FP  # 被模型预测为负类的负样本

            elif len_groundtruth != 0 and len_simulate == 0:
                P = len_groundtruth
                N = N_plus_P - P
                TP = 0
                FP = 0
                FN = P
                TN = N

            elif len_groundtruth != 0 and len_simulate != 0:

                groundtruth_heavy_hitter_set = groundtruth_epoch_group.get_group(epoch_i)
                #groundtruth_heavy_hitter_set.drop_duplicates(['id'], keep='first', inplace=True)
                # groundtruth_heavy_hitter_set = groundtruth_heavy_hitter_set[['epoch', 'id', 'packets_in_flow', 'global_T']]

                simulating_heavy_hitter_set = simulate_epoch_group.get_group(epoch_i)
                #simulating_heavy_hitter_set.drop_duplicates(['id'], keep='first', inplace=True)
                # simulating_heavy_hitter_set = simulating_heavy_hitter_set[['epoch', 'id', 'packets_num', 'global_threshold']]
                P = len_groundtruth
                N = N_plus_P - P

                for index, row in simulating_heavy_hitter_set.iterrows():
                    flow_ID = row['id']
                    if flow_ID in groundtruth_heavy_hitter_set.values:
                        TP += 1  # 被模型预测为正类的正样本
                    else:
                        FP += 1  # 被模型预测为正类的负样本

                for index, row in groundtruth_heavy_hitter_set.iterrows():
                    flow_ID = row['id']
                    if flow_ID not in simulating_heavy_hitter_set.values:
                        FN += 1  # 被模型预测为负类的正样本

                TN = N - FP
            
            logging.info("N = {}; P = {}; TP = {}; FP = {}; TN = {}; FN = {};".format(N, P, TP, FP, TN, FN))
            acc = (TP + TN) / (P + N)
            # precision = TP / (TP + FP)
            # recall = TP / (TP + FN)

            if TP + FP == 0:
                precision = 1.0

            else:
                precision = TP / (TP + FP)

            if TP + FN == 0:
                recall = 1.0

            else:
                recall = TP / (TP + FN)

            f1 = 2.0 * (precision * recall) / (precision + recall)
            logging.info("acc = {}; pre = {}; rec = {}; f1 = {}".format(acc, precision, recall, f1))
            result_df = result_df.append([{"epoch": epoch_i,
                                           "N": N_plus_P - P,
                                           "P": P,
                                           "TP": TP,
                                           "FP": FP,
                                           "TN": TN,
                                           "FN": FN,
                                           "acc": acc,
                                           "precision": precision,
                                           "recall": recall,
                                           "f1": f1}], ignore_index=True)
        if len(result_df) != 0:
            N_total = result_df['N'].sum()
            P_total = result_df['P'].sum()
            TP_total = result_df['TP'].sum()
            FP_total = result_df['FP'].sum()
            TN_total = result_df['TN'].sum()
            FN_total = result_df['FN'].sum()
            acc_total = round((TP_total + TN_total) / (P_total + N_total), 5)
            pre_total = round(TP_total / (TP_total + FP_total), 5)
            rec_total = round(TP_total / (TP_total + FN_total), 5)
            f1_total = round(2.0 * (pre_total * rec_total) / (pre_total + rec_total), 5)
            result_df = result_df.append([{"epoch": "mean",
                                           "N": N_total,
                                           "P": P_total,
                                           "TP": TP_total,
                                           "FP": FP_total,
                                           "TN": TN_total,
                                           "FN": FN_total,
                                           "acc": acc_total,
                                           "precision": pre_total,
                                           "recall": rec_total,
                                           "f1": f1_total}], ignore_index=True)
        logging.info("############################## all ##############################".format(epoch_i, len_groundtruth, len_simulate))
        logging.info("MEAN: acc = {}; pre = {}; rec = {}; f1 = {}".format(result_df['acc'].mean(), result_df['precision'].mean(), result_df['recall'].mean(), result_df['f1'].mean()))

        logging.info("N = {}; P = {}; TP = {}; FP = {}; TN = {}; FN = {};".format(N_total, P_total, TP_total, FP_total, TN_total, FN_total))
        logging.info("acc = {}; pre = {}; rec = {}; f1 = {}".format(acc_total, pre_total, rec_total, f1_total))

        result_df.to_csv(save_data_dir + "[analyse F1].csv")


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
    evaluation_file_name = save_data_dir + "[evaluate predict].csv"
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

def evaluate_error():
    global algorithm
    error_df = pd.read_csv(save_data_dir + "[volume estimation].csv")
    mean_error = error_df['error%'].mean()
    error_df['error%'] = error_df['error%'].abs()
    abs_mean_error = error_df['error%'].mean()

    error_file_name = save_data_dir + "error_{}_{}.txt".format(algorithm, round(abs_mean_error, 4))
    file = open(error_file_name, "w")
    file.write("abs {}_error={}".format(algorithm, abs_mean_error))
    file.write("mean {}_error={}".format(algorithm, mean_error))
    print(f"estimated volume abs mean error% = {abs_mean_error}, estimated volume mean error% = {mean_error}")

if "__main__" == __name__:
    logging.getLogger().setLevel(logging.INFO)

    dataset = "univ1"
    file_begin = 1
    file_end = 20
    time_interval = 65
    theta = 0.01
    x = 6000
    fattree_k = 4
    alpha = 1
    heap_num = 2
    algorithm = "kmv"
    # dataset = sys.argv[1]       
    # file_begin = int(sys.argv[2])
    # file_end = int(sys.argv[3])
    # time_interval = int(sys.argv[4])
    # fattree_k = int(sys.argv[5])
    # alpha = int(sys.argv[6])
    # theta = float(sys.argv[7])
    # x = int(sys.argv[8])
    # algorithm = sys.argv[9]

    save_data_dir = ""
    if algorithm == "kmv":
        save_data_dir = "{}_trace/{}_[{}-{}]_epoch{}_fattree{}/x{}_theta{}_{}_alpha{}/".format(dataset, dataset, file_begin, file_end, time_interval, fattree_k, x, theta, algorithm, alpha)
    elif algorithm == "multi":
        save_data_dir = "{}_trace/{}_[{}-{}]_epoch{}_fattree{}/x{}_theta{}_{}_heap{}/".format(dataset, dataset, file_begin, file_end, time_interval, fattree_k, x, theta, algorithm, heap_num)
    else: 
        save_data_dir = "{}_trace/{}_[{}-{}]_epoch{}_fattree{}/x{}_theta{}_{}/".format(dataset, dataset, file_begin, file_end, time_interval, fattree_k, x, theta, algorithm)

    groundtruth_dir =  "{}_trace/{}_[{}-{}]_epoch{}_fattree{}/groundtruth_HH_theta{}.csv".format(dataset, dataset, file_begin, file_end, time_interval, fattree_k, theta)
    simulating_dir = save_data_dir + "[heavy hitter].csv"
    
    evaluate_HH(groundtruth_dir, simulating_dir)

    if algorithm == "lstm" or algorithm == "ewma" :
        evaluate_predict_hash(save_data_dir + "[hash track].csv")

    evaluate_error()


