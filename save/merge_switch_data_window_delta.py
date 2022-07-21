from bdb import effective
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
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from scipy import stats

class programmableSwitch:

    def __init__(self, switch_id):
        self.switch_id = switch_id
        self.p_threshold = 0.3
        self.packets_through = pd.DataFrame()
        self.packets_sampled = pd.DataFrame()
        self.sample_max_hash = 0.0
        self.delta = 1.0
    
    def initializeSwitch(self, epoch_i):
        global switch_data_dict
        self.packets_through = switch_data_dict[self.switch_id].get_group(epoch_i)
    
    def sample_packets(self):
        global x, safe_x
        self.packets_through.sort_values(by="hash", inplace=True, ascending=True)
        self.packets_sampled = self.packets_through.head(safe_x)
        self.packets_sampled =  self.packets_sampled.reset_index(drop=True)
        self.p_threshold = self.packets_sampled.at[len(self.packets_sampled)-1, 'hash']
        self.delta = self.p_threshold / len(self.packets_sampled) * len(self.packets_through)
        hash_data = [[epoch_i, self.switch_id, len(self.packets_through), len(self.packets_sampled), self.p_threshold]]
        with open(hash_file_name, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(hash_data)


class physicalNetwork:

    def __init__(self, fattree_k):
        self.fattree_k = fattree_k
        self.networkSwitches = {}

        self.all_sampled_packet = pd.DataFrame()
        self.effective_x = 0  
        self.effective_x_packet = pd.DataFrame()   

        self.estimated_volume = 0
        self.delta_estimated_volume = 0 
        self.global_sampling_rate = 0.0

    def initializeNetwork(self, epoch_i):
        global switch_num
        logging.info("###################################### epoch {} ######################################".format(epoch_i))
        for switch_id in range(switch_num):
            self.networkSwitches[switch_id].initializeSwitch(epoch_i)  # 打数据包
            self.networkSwitches[switch_id].sample_packets()    # 采样数据包
        
    def merge_data(self):

        self.all_sampled_packet = pd.DataFrame()
        self.all_sampled_packet['switch_id'] = ''

        for switch_id in range(switch_num):

            # collect data
            packets_switch_sampled = self.networkSwitches[switch_id].packets_sampled
            packets_switch_sampled['switch_id'] = switch_id
            self.all_sampled_packet = self.all_sampled_packet.append(packets_switch_sampled)

            # effective x
            if self.effective_x == 0:
                self.effective_x = len(packets_switch_sampled)
            else:
                self.effective_x = min(self.effective_x, len(packets_switch_sampled))

        # merge all
        before_merge_len = len(self.all_sampled_packet)
        self.all_sampled_packet = pd.DataFrame(self.all_sampled_packet)
        self.all_sampled_packet.drop_duplicates(subset=['new_id'], keep='first', inplace=True)
        after_merge = len(self.all_sampled_packet)

        logging.info("Merge {} packets to {} packets; every switch get top {} packets".format(before_merge_len, after_merge, self.effective_x))

        # top safe_x hash
        self.all_sampled_packet.sort_values(by="hash", inplace=True, ascending=True)
        self.all_sampled_packet.drop_duplicates(subset=['hash'], keep='first', inplace=True)
        self.effective_x_packet = self.all_sampled_packet.head(self.effective_x)
        self.effective_x_packet = self.effective_x_packet.reset_index()

    def estimate_volume(self):
        global V_file_name, epoch_i, epoch_i, x

        self.all_sampled_packet =  self.all_sampled_packet.reset_index(drop=True)
        should_x_hash = self.all_sampled_packet.at[x-1, 'hash']
        actual_volume = volume_groundtruth.at[epoch_i, 'actual_volume']

        # window
        if self.effective_x > x:
            estimated_volume_list = []
            for window_left in range(0, self.effective_x - x + 1):
                window_right = window_left + x - 1
                h  = self.effective_x_packet.at[window_right, 'hash'] - self.effective_x_packet.at[window_left, 'hash']
                k  = x
                estimated_volume_list.append(int((k - 1) / h))
            self.estimated_volume = stats.hmean(estimated_volume_list)
        else:
            h = self.effective_x_packet['hash'].max()
            k = len(self.effective_x_packet)

            self.estimated_volume = int((k - 1) / h)
        
        # delta : harmonic mean 
        delta_list = []
        for switch_id in range(switch_num):
            delta_list.append(self.networkSwitches[switch_id].delta)
        delta = stats.hmean(delta_list)
        # delta = np.mean(delta_list)
        logging.info("【TASK1】delta_list = {}".format(delta_list))
        self.delta_estimated_volume = int(((k - 1) / h) * delta)

        actual_volume = volume_groundtruth.at[epoch_i, 'actual_volume']
        error = (int(self.estimated_volume) - int(actual_volume)) / actual_volume
        delta_error = (int(self.delta_estimated_volume) - int(actual_volume)) / actual_volume
        logging.info("【TASK1】estimated volume = {}; actual volume = {}, ERROR = {}; effectice_sampled_packets_num = {}".format(self.estimated_volume, actual_volume, 100 * error, self.effective_x))
        logging.info("【TASK1】delta estimated volume = {}, delta = {}; actual volume = {}, ERROR = {}; ".format(self.delta_estimated_volume, delta, actual_volume, 100 * delta_error))

        # write HH track file
        V_data = [[epoch_i, actual_volume, self.estimated_volume, 100 * error, delta, self.delta_estimated_volume, 100 * delta_error]]
        with open(V_file_name, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(V_data)
    
    def check_heavy_hitter(self):
        global epoch_i, theta, HH_file_name, epoch_i

        # calculate sample global sampling rate
        self.global_sampling_rate = self.effective_x / self.estimated_volume
        global_threshold = int(self.estimated_volume * theta)
        sample_global_threshold = int(self.estimated_volume * theta) * self.global_sampling_rate

        # CM sketch take a count
        CM_sketch = {}
        for index, row in self.effective_x_packet.iterrows():
            flow_id = row['id']
            if flow_id in CM_sketch.keys():
                CM_sketch[flow_id] += 1
            else:
                CM_sketch[flow_id] = 1

        # check heavy hitter
        for flow_id, counter in CM_sketch.items():

            if CM_sketch[flow_id] > sample_global_threshold:

                # write HH track file
                HH_data = [[epoch_i, flow_id, self.effective_x, self.global_sampling_rate, global_threshold, sample_global_threshold, counter / self.global_sampling_rate, counter]]
                with open(HH_file_name, mode="a", newline="") as file_:
                    writer = csv.writer(file_)
                    writer.writerows(HH_data)

        logging.info("【TASK2】HH DETECTION: global_sampling_rate = {}, theta = {}, global_threshold = {} ".format(self.global_sampling_rate, theta, global_threshold))

    def reset_network(self):
        # reset switches
        for switch_id in self.networkSwitches.keys():
            self.networkSwitches[switch_id].packets_through = pd.DataFrame()
            self.networkSwitches[switch_id].packets_sampled = pd.DataFrame()
            self.networkSwitches[switch_id].sample_max_hash = 0.0

        # reset central controller
        self.all_sampled_packet = pd.DataFrame()
        self.effective_x = 0
        self.effective_x_packet = pd.DataFrame()

        self.estimated_volume = 0
        self.global_sampling_rate = 0.0

def simulate(fattree_k):

    global switch_num, epoch_len, epoch_i

    network = physicalNetwork(fattree_k)
    for switch_id in range(switch_num):
        network.networkSwitches[switch_id] = programmableSwitch(switch_id)

    logging.info("{} epoches in all".format(epoch_len+1))
    for epoch_i in range(epoch_len+1):
        network.initializeNetwork(epoch_i)
        network.merge_data()
        network.estimate_volume()
        network.check_heavy_hitter()
        network.reset_network()

def set_res_file():
    global hash_file_name, HH_file_name, V_file_name
    """
    hash track file
    """ 
    hash_file_name = save_data_dir + "[hash track].csv"
    header = ["epoch","switch_id", "len_packets_through","len_packets_sampled",  "should_max_hash"]
    if os.path.exists(hash_file_name):
        os.remove(hash_file_name)
    os.mknod(hash_file_name)

    logging.info("{} set already! ".format(hash_file_name))

    with open(hash_file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

    """
    heavy hitter track file
    """
    HH_file_name = save_data_dir + "[heavy hitter].csv"
    header = ["epoch", "id", "effective_x", "sampling_rate", "global_threshold",  "sample_global_threshold", "global_packets_num", "packets_num"]

    if os.path.exists(HH_file_name):
        os.remove(HH_file_name)
    os.mknod(HH_file_name)

    logging.info("{} set already! ".format(HH_file_name))

    with open(HH_file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

    """
    volume track file
    """
    V_file_name = save_data_dir + "[volume estimation].csv"
    header = ["epoch", "actual_volume", "estimated_volume", "error%", "delta", "delta_estimated_volume", "delta_error%"]

    if os.path.exists(V_file_name):
        os.remove(V_file_name)
    
    os.mknod(V_file_name)

    logging.info("{} set already! ".format(V_file_name))

    with open(V_file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    dataset = 'univ1'
    file_begin = 1
    file_end = 20
    time_interval = 65
    fattree_k = 4
    x = 6000
    theta = 0.01

    # dataset = sys.argv[1]       
    # file_begin = int(sys.argv[2])
    # file_end = int(sys.argv[3])
    # time_interval = int(sys.argv[4])
    # fattree_k = int(sys.argv[5])
    # theta = float(sys.argv[6])
    # x = int(sys.argv[7])

    alpha = 1.1
    safe_x = int(x * alpha)

    len_core = int(pow(fattree_k / 2, 2))
    len_aggr = int(fattree_k / 2)
    len_edge = int(fattree_k / 2)
    switch_num = len_core + (len_aggr + len_edge) * fattree_k

    logging.info("【window_delta】: time interval = {}, fattree_k = {}, x = {}, theta = {} over".format(time_interval, fattree_k, x, theta))

    # read switch data
    switch_data_dir = "univ1_trace/univ1_[{}-{}]_epoch{}_fattree{}/".format(file_begin, file_end, time_interval, fattree_k)
    switch_data_dict = {}
    
    for switch_id in range(switch_num):

        single_switch_data = pd.DataFrame(pd.read_csv(switch_data_dir + "/switch_data/switch{}.csv".format(switch_id)))
        logging.info("getting switch{} data...{} packets".format(switch_id, len(single_switch_data)))
        single_switch_data['seq'] = single_switch_data['seq'].apply(str)
        # single_switch_data.drop_duplicates(subset=['new_id'], keep='first', inplace=True)

        grouped_switch_data = single_switch_data.groupby('epoch')
        switch_data_dict[switch_id] = grouped_switch_data

        logging.info("over")

    count_file_dir = switch_data_dir + "switch_counter.csv"
    epoch_len = pd.read_csv(count_file_dir)['epoch'].max()

    # save dir
    save_data_dir = "univ1_trace/univ1_[{}-{}]_epoch{}_fattree{}/x{}_theta{}_window_delta/".format(file_begin, file_end, time_interval, fattree_k, x, theta)
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    # read groundtruth data
    volume_groundtruth = pd.read_csv(switch_data_dir + "groundtruth_volume.csv")

    # res file
    set_res_file()

    # simulate
    simulate(fattree_k=fattree_k)
    logging.info("【window_delta】: time interval = {}, fattree_k = {}, x = {}, theta = {} over".format(time_interval, fattree_k, x, theta))








