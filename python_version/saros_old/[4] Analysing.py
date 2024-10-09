import pandas as pd
import sys
import logging
import numpy as np

if "__main__" == __name__:
    logging.getLogger().setLevel(logging.INFO)

    file_begin = 1
    file_end = 20
    time_interval = 10
    theta = 0.01
    x = 1000

    groundtruth_dir =  "univ1_trace/univ1_[{}-{}]_epoch{}_[groundtruth]_theta{}.csv".format(file_begin, file_end, time_interval, theta)
    simulating_dir = "univ1_trace/univ1_[{}-{}]_epoch{}_[heavy hitter]_x{}_theta{}.csv".format(file_begin, file_end, time_interval, x, theta)

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
            logging.info("N_plus_P={}".format(N_plus_P))
            
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

        result_df.to_csv("univ1_trace/univ1_[{}-{}]_epoch{}_[analyse F1]_theta{}.csv".format(file_begin, file_end, time_interval, theta))
