from click import group
import pandas as pd
import numpy as np
import random
import logging
import sys
import csv
import os
import shutil
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def set_file(should_hash_file):

    header = ["epoch", "len_packets_through", "should_max_hash", "should_max_hash_diff"]

    if os.path.exists(should_hash_file):
        os.remove(should_hash_file)
    os.mknod(should_hash_file)

    logging.info("{} set already! ".format(should_hash_file))

    with open(should_hash_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    file_begin = 1
    file_end = 20
    time_interval = 45
    fattree_k = 4

    x = 500
    theta = 0.01
    alpha = 0.8

    len_core = int(pow(fattree_k / 2, 2))
    len_aggr = int(fattree_k / 2)
    len_edge = int(fattree_k / 2)
    switch_num = len_core + (len_aggr + len_edge) * fattree_k

    # read switch data
    switch_data_dir = "univ1_trace/univ1_[{}-{}]_epoch{}_fattree{}/switch_data/".format(file_begin, file_end, time_interval, fattree_k)
    save_data_dir = "univ1_trace/univ1_[{}-{}]_epoch{}_fattree{}/x{}_should_max_hash/".format(file_begin, file_end, time_interval, fattree_k, x)
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    for switch_id in range(switch_num):
        should_hash_file = save_data_dir + "switch{}.csv".format(switch_id)
        set_file(should_hash_file)

        logging.info("getting switch{} max hash...".format(switch_id))
        single_switch_data = pd.DataFrame(pd.read_csv(switch_data_dir + "switch{}.csv".format(switch_id)))
        grouped_switch_data = single_switch_data.groupby('epoch')

        pre_should_max_hash = 0
        for group in grouped_switch_data:
            epoch_num = group[0]
            epoch_packet = group[1]

            epoch_packet =  epoch_packet.reset_index(drop=True)
            epoch_packet.sort_values(by="hash", inplace=True, ascending=True)

            if len(epoch_packet) >= x:
                should_max_hash = epoch_packet.hash.nsmallest(x).iloc[-1]# self.packets_through.at[sample_num-1, 'hash'] # 改了这里变成sample_num - 1
            else:
                should_max_hash = -1

            hash_data = [[epoch_num, len(epoch_packet), should_max_hash, should_max_hash - pre_should_max_hash]]
            with open(should_hash_file, mode="a", newline="") as file_:
                writer = csv.writer(file_)
                writer.writerows(hash_data)
            
            pre_should_max_hash = should_max_hash
            # if not os.path.exists(save_data_dir + "switch{}/".format(switch_id)):
            #     os.makedirs(save_data_dir + "switch{}/".format(switch_id))
            # epoch_packet.to_csv(save_data_dir + "switch{}/epoch{}_{}.csv".format(switch_id, epoch_num, round(should_max_hash, 3)))
