import pandas as pd
import json
import numpy as np
import time, datetime
from datetime import date, datetime, timedelta
import sys
import logging
import os
import csv

def generate_groundtruth(raw_df, theta):

    CM_c = {}
    Heavy_Hitter_Set = pd.DataFrame(columns=['epoch', 'packets_in_epoch', "theta",'global_T', 'id', 'packets_in_flow'])
    grouped_packet = raw_df.groupby('epoch')

    # for every epoch
    for group in grouped_packet:

        epoch_num = group[0]
        packet = group[1]
        pre_len = len(packet)

        # packets in epoch
        packet.drop_duplicates(subset=['new_id'], keep='first', inplace=True)
        # logging.info("epoch{}, {} packets --> {} no-duplicate packets".format(epoch_num, pre_len, len(packet)))

        V_data = [[epoch_num, len(packet)]]
        with open(V_file_name, mode="a", newline="") as file_:
            writer = csv.writer(file_)
            writer.writerows(V_data)
        
        # global threshold
        global_threshold = len(packet) * theta

        # check heavy hitter
        count_every_flow = packet.groupby('id')
        # logging.info("has {} different flows".format(len(count_every_flow)))
        for flow in count_every_flow:
            flow_id = flow[0]
            packets_in_flow = flow[1]
            
            if len(packets_in_flow) >= global_threshold:
                logging.info("epoch{}, global_T = {}, heavy hitter:{}".format(epoch_num, global_threshold, flow_id))
                Heavy_Hitter_Set = Heavy_Hitter_Set.append([{"epoch": epoch_num,
                                                             "packets_in_epoch": len(packet),
                                                             "theta": theta, 
                                                             "global_T": global_threshold,
                                                             "id": flow_id,
                                                             "packets_in_flow": len(packets_in_flow),
                                                             }
                                                            ], ignore_index=True)

        logging.info("epoch {} end; global_threshold = {}".format(epoch_num, global_threshold))

    Heavy_Hitter_Set.to_csv(save_data_dir + "groundtruth_HH_theta{}.csv".format(theta))
    logging.info("has {} heavy hitter flow".format(len(Heavy_Hitter_Set)))
    return Heavy_Hitter_Set


if "__main__" == __name__:
    logging.getLogger().setLevel(logging.INFO)

    time_interval = 75
    theta = 0.01
    fattree_k = 4

    file_begin = 1
    file_end = 20

    # read raw packets
    raw_df = pd.read_csv("univ1_trace/data/univ1_[{}-{}]_epoch{}.csv".format(file_begin, file_end, time_interval))
    raw_df = raw_df[['id', 'epoch', 'seq', 'new_id', 'hash']]
    raw_df.info()

    save_data_dir = "univ1_trace/univ1_[{}-{}]_epoch{}_fattree{}/".format(file_begin, file_end, time_interval, fattree_k)
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    """
    volume groundtruth
    """
    V_file_name = save_data_dir + "groundtruth_volume.csv".format(file_begin, file_end, time_interval, fattree_k)
    header = ["epoch", "actual_volume"]

    if os.path.exists(V_file_name):
        os.remove(V_file_name)
    
    os.mknod(V_file_name)
    with open(V_file_name, mode="a", newline="") as counter_file:
        writer = csv.writer(counter_file)
        writer.writerow(header)

    # """
    # HH groundtruth
    # """
    heavy_hitter = generate_groundtruth(raw_df, theta)
    heavy_hitter.info()

    raw_df.info()

