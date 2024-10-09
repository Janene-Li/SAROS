import pandas as pd
import numpy as np
import random
import logging
import sys
import csv
import os
from scipy import stats
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class programmableSwitch:

    def __init__(self, switch_id):
        self.switch_id = switch_id
        self.packets_through = pd.DataFrame()
        self.packets_sampled = [[2.0, "", ""]] * int(x * alpha)

    def initializeSwitch(self, epoch_i):
        global switch_data_dict
        self.packets_through = switch_data_dict[self.switch_id].get_group(epoch_i)
        self.packets_sampled = [[2.0, "", ""]] * int(x * alpha)

    def sample_packets(self):
        global x
        for index, row in self.packets_through.iterrows():
            random.seed(row['new_id'])
            first_hash_value = random.randint(0,x)
            second_hash_value = row['hash']
            if self.packets_sampled[first_hash_value][0] > second_hash_value:
                self.packets_sampled[first_hash_value][0] = second_hash_value
                self.packets_sampled[first_hash_value][1] = row['new_id']
                self.packets_sampled[first_hash_value][2] = row['id']

class physicalNetwork:

    def __init__(self, fattree_k):
        self.fattree_k = fattree_k
        self.networkSwitches = {}
        self.all_sampled_packet = [[2.0, "", ""]] * int(x * alpha)
        self.effective_x = 0
        self.harmonic_estimated_volume = 0.0
        self.global_threshold = 0.0
        self.global_sampling_rate = 0.0

    def initializeController(self, switch_num):
        for switch_id in range(switch_num):
            self.networkSwitches[switch_id] = programmableSwitch(switch_id)

    def initializeNetwork(self, epoch_i):
        global switch_num

        logging.info("###################################### epoch {} ######################################".format(epoch_i))
        for switch_id in range(switch_num):
            self.networkSwitches[switch_id].initializeSwitch(epoch_i)  # 打数据包
            self.networkSwitches[switch_id].sample_packets()    # 采样数据包

    def merge_data(self):
        global x, alpha

        self.all_sampled_packet = [[2.0, ""]] * int(x * alpha)
        for switch_id in range(switch_num):
            for i in range(0, int(x*alpha)):
                if self.networkSwitches[switch_id].packets_sampled[i][0] < self.all_sampled_packet[i][0]:
                    self.all_sampled_packet[i] = self.networkSwitches[switch_id].packets_sampled[i]
            logging.info("【TASK0】merge switch {} already".format(switch_id))
            
    def estimate_volume(self):
        global x, alpha, theta, V_file_name, epoch_i

        volume_estimate_list = []
        for i in range(0, int(x*alpha)):
            if self.all_sampled_packet[i][0] < 2.0:
                estimated_volume_i = 1/self.all_sampled_packet[i][0]
                volume_estimate_list.append(estimated_volume_i)

        self.effective_x = len(volume_estimate_list)
        self.harmonic_estimated_volume = stats.hmean(volume_estimate_list)

        actual_volume = volume_groundtruth.at[epoch_i, 'actual_volume']
        error = (int(self.harmonic_estimated_volume) - int(actual_volume)) / int(actual_volume)
        
        self.global_threshold = self.harmonic_estimated_volume * theta
        self.global_sampling_rate = x / self.global_threshold

        # write HH track file
        V_data = [[epoch_i, actual_volume, self.harmonic_estimated_volume, 100 * error]]
        with open(V_file_name, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(V_data)

        logging.info("【TASK1】VOLUME ESTIMATION = {}, actual volume = {}, ERROR = {}; effectice_sampled_packets_num = {}; global threshold = {}, global sampling rate = {}".format(self.harmonic_estimated_volume, actual_volume, 100 * error, self.effective_x, self.global_threshold, self.global_sampling_rate))


    def check_heavy_hitter(self):
        global epoch_i, theta, HH_file_name

        # CM sketch take a count
        CM_sketch = {}
        for i in range(0, int(x*alpha)):
            flow_id = self.all_sampled_packet[i][2]
            if flow_id in CM_sketch.keys():
                CM_sketch[flow_id] += 1
            else:
                CM_sketch[flow_id] = 1

        # check heavy hitter
        sample_global_threshold = self.global_threshold * self.global_sampling_rate
        for flow_id, counter in CM_sketch.items():

            if CM_sketch[flow_id] > sample_global_threshold:

                # write HH track file
                HH_data = [[epoch_i, flow_id, self.effective_x, self.global_sampling_rate, self.global_threshold, sample_global_threshold, counter / self.global_sampling_rate, counter]]
                with open(HH_file_name, mode="a", newline="") as file_:
                    writer = csv.writer(file_)
                    writer.writerows(HH_data)

        logging.info("【TASK2】HH DETECTION: global_sampling_rate = {}, theta = {}, global_threshold = {} ".format(self.global_sampling_rate, theta, self.global_threshold))

    def reset_network(self):
        # reset switches
        for switch_id in self.networkSwitches.keys():
            self.networkSwitches[switch_id].packets_through = pd.DataFrame()
            self.networkSwitches[switch_id].packets_sampled = [[1.0, "", ""]] * int(x * alpha)

        # reset central controller
        self.all_sampled_packet = [[1.0, "", ""]] * int(x * alpha)
        self.harmonic_estimated_volume = 0.0
        self.global_threshold = 0.0
        self.global_sampling_rate = 0.0
        self.effective_x = 0

def simulate(fattree_k):
    global switch_num, epoch_len, epoch_i

    network = physicalNetwork(fattree_k)
    network.initializeController(switch_num)
    
    logging.info("({} epoches in all".format(epoch_len+1))
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
    header = ["epoch","switch_id", "len_packets_through","len_packets_sampled", "x", "sample_max_hash", "should_max_hash", "should-actual", "predict_max_hash"]

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
    header = ["epoch", "actual_volume", "estimated_volume", "error%"]

    if os.path.exists(V_file_name):
        os.remove(V_file_name)
    
    os.mknod(V_file_name)

    logging.info("{} set already! ".format(V_file_name))

    with open(V_file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    global x, alpha, theta, fattree_k, time_interval

    dataset = "univ1"
    file_begin = 1
    file_end = 20
    time_interval = 30

    x = 1000
    theta = 0.01
    alpha = 1.5

    fattree_k = 4
    len_core = int(pow(fattree_k / 2, 2))
    len_aggr = int(fattree_k / 2)
    len_edge = int(fattree_k / 2)
    switch_num = len_core + (len_aggr + len_edge) * fattree_k

    # read switch data
    switch_data_dir = "{}_trace/{}_[{}-{}]_epoch{}_fattree{}/".format(dataset, dataset, file_begin, file_end, time_interval, fattree_k)
    switch_data_dict = {}

    for switch_id in range(switch_num):

        single_switch_data = pd.DataFrame(pd.read_csv(switch_data_dir + "/switch_data/switch{}.csv".format(switch_id)))
        logging.info("getting switch{} data...{} packets".format(switch_id, len(single_switch_data)))
        single_switch_data['seq'] = single_switch_data['seq'].apply(str)

        grouped_switch_data = single_switch_data.groupby('epoch')
        switch_data_dict[switch_id] = grouped_switch_data

        logging.info("over")

    count_file_dir = switch_data_dir + "switch_counter.csv"
    epoch_len = pd.read_csv(count_file_dir)['epoch'].max()

    # save dir
    save_data_dir = "{}_trace/{}_[{}-{}]_epoch{}_fattree{}/x{}_theta{}_kmv/".format(dataset, dataset, file_begin, file_end, time_interval, fattree_k, x, theta)
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    # read groundtruth data
    volume_groundtruth = pd.read_csv(switch_data_dir + "groundtruth_volume.csv")

    # res file
    set_res_file()

    # simulate
    simulate(fattree_k=fattree_k)

    logging.info("new: time interval = {}, fattree_k = {}, x = {}, theta = {} over".format(time_interval, fattree_k, x, theta))


