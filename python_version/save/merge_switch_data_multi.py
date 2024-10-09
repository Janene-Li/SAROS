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
        self.packets_through = pd.DataFrame()
        self.packets_sampled = {}
        self.sample_max_hash = 0.0
    
    def initializeSwitch(self, epoch_i):
        global switch_data_dict
        self.packets_through = switch_data_dict[self.switch_id].get_group(epoch_i)
        self.packets_through = self.packets_through.sort_values(by="hash", ascending=True)
    
    def sample_packets(self):
        global x, heap_num
        self.packets_sampled = {}
        for heap_id in range(heap_num):
            range_left = 1.0/(heap_num) * (heap_id)
            range_right = (1.0/heap_num) * (heap_id + 1)
            heap_x = int(x/heap_num)
            heap_packets_sampled = self.packets_through[(range_right > self.packets_through['hash']) & (range_left < self.packets_through['hash'])]
            heap_packets_sampled = heap_packets_sampled.head(heap_x)

            self.packets_sampled[heap_id] = heap_packets_sampled
            logging.info("【sample】swicth{}, heap{}, {} packets;".format(self.switch_id, heap_id, len(heap_packets_sampled)))

        hash_data = [[epoch_i, self.switch_id, len(self.packets_through), len(self.packets_sampled)]]
        with open(hash_file_name, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(hash_data)


class physicalNetwork:

    def __init__(self, fattree_k):
        self.fattree_k = fattree_k
        self.networkSwitches = {}

        self.all_sampled_packet = {}
        self.effective_x = 0
        self.effective_x_packet = pd.DataFrame()
        self.estimated_volume = 0
        self.global_sampling_rate = 0.0

    def initializeNetwork(self, epoch_i):
        global switch_num
        logging.info("###################################### epoch {} ######################################".format(epoch_i))
        for switch_id in range(switch_num):
            self.networkSwitches[switch_id].initializeSwitch(epoch_i)  # 打数据包
            self.networkSwitches[switch_id].sample_packets()    # 采样数据包
        
    def merge_data(self):

        self.all_sampled_packet = {}

        for heap_id in range(heap_num):
            heap_x = int(x/heap_num)

            for switch_id in range(switch_num): 
                if heap_id not in self.all_sampled_packet.keys():
                    self.all_sampled_packet[heap_id] = pd.DataFrame()
                self.all_sampled_packet[heap_id] = self.all_sampled_packet[heap_id].append(self.networkSwitches[switch_id].packets_sampled[heap_id])
            
            before_merge_len = len(self.all_sampled_packet[heap_id])

            # merge
            self.all_sampled_packet[heap_id].drop_duplicates(subset=['new_id'], keep='first', inplace=True)
            self.all_sampled_packet[heap_id] = self.all_sampled_packet[heap_id].reset_index()
            self.all_sampled_packet[heap_id] = self.all_sampled_packet[heap_id].sort_values(by="hash", ascending=True)
            
            after_merge_len = len(self.all_sampled_packet[heap_id])

            if after_merge_len > heap_x:
                self.all_sampled_packet[heap_id] = self.all_sampled_packet[heap_id].head(heap_x)
                after_merge_len = len(self.all_sampled_packet[heap_id])

            # self.effective_x += len(self.all_sampled_packet[heap_id])
            self.effective_x_packet = self.effective_x_packet.append(self.all_sampled_packet[heap_id])

            logging.info("【TASK0】heap{}: Merge {} packets to {} packets;".format(heap_id, before_merge_len, after_merge_len))
        self.effective_x = len(self.effective_x_packet)
        logging.info("【TASK0】effective sample {} packets;".format(self.effective_x))

    def estimate_volume(self):
        global V_file_name, epoch_i, epoch_i, heap_num

        heap_estimated_volume = []
        heap_max_hash = []

        for heap_id in range(heap_num):
  
            heap_x = int(x/heap_num)
            range_left = 1.0/(heap_num) * (heap_id)

            # max hash
            heap_sampled_packets = self.all_sampled_packet[heap_id]
            heap_sampled_packets = heap_sampled_packets.reset_index()

            if len(heap_sampled_packets) < heap_x:
                heap_x = len(heap_sampled_packets)
            max_hash = heap_sampled_packets.at[heap_x-1, 'hash']
            
            # estimate volume
            heap_max_hash.append(max_hash)
            estimated_volume = int( heap_x / (max_hash - range_left) / heap_num )
            # estimated_volume = int(len(heap_sampled_packets) * (1 / heap_num) / (max_hash - range_left))

            heap_estimated_volume.append(estimated_volume)
            logging.info("【TASK1】heap{}: VOLUME ESTIMATION = {}, max_hash - range_left = {}, heap_x = {}".format(heap_id, estimated_volume,  (max_hash - range_left), heap_x))

        self.estimated_volume = stats.hmean(heap_estimated_volume) * heap_num
        actual_volume = volume_groundtruth.at[epoch_i, 'actual_volume']
        error = (int(self.estimated_volume) - int(actual_volume)) / actual_volume
        logging.info("【TASK1】VOLUME ESTIMATION = {}, actual volume = {}, ERROR = {}; ".format(self.estimated_volume, actual_volume, 100 * error))

        # write HH track file
        V_data = [[epoch_i, actual_volume, self.estimated_volume, 100 * error]]
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
        self.effective_x_packet  = self.effective_x_packet.reset_index()
        for index, row in self.effective_x_packet.iterrows():
            flow_id = row['id']
            if flow_id in CM_sketch.keys():
                CM_sketch[flow_id] += 1
            else:
                CM_sketch[flow_id] = 1

        # check heavy hitter
        HH_counter = 0
        for flow_id, counter in CM_sketch.items():

            if CM_sketch[flow_id] > sample_global_threshold:
                HH_counter += 1
                # write HH track file
                HH_data = [[epoch_i, flow_id, self.effective_x, self.global_sampling_rate, global_threshold, sample_global_threshold, counter / self.global_sampling_rate, counter]]
                with open(HH_file_name, mode="a", newline="") as file_:
                    writer = csv.writer(file_)
                    writer.writerows(HH_data)

        logging.info("【TASK2】HH DETECTION: global_sampling_rate = {}, theta = {}, global_threshold = {}; HH num = {}".format(self.global_sampling_rate, theta, global_threshold, HH_counter))

    def reset_network(self):
        # reset switches
        for switch_id in self.networkSwitches.keys():
            self.networkSwitches[switch_id].packets_through = pd.DataFrame()
            self.networkSwitches[switch_id].packets_sampled = pd.DataFrame()
            self.networkSwitches[switch_id].sample_max_hash = 0.0

        # reset central controller
        self.all_sampled_packet = {}
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

    dataset = 'univ1'
    file_begin = 1
    file_end = 20
    time_interval = 65
    fattree_k = 4
    x = 6000
    theta = 0.01
    heap_num = 2

    # dataset = sys.argv[1]       
    # file_begin = int(sys.argv[2])
    # file_end = int(sys.argv[3])
    # time_interval = int(sys.argv[4])
    # fattree_k = int(sys.argv[5])
    # theta = float(sys.argv[6])
    # x = int(sys.argv[7])

    len_core = int(pow(fattree_k / 2, 2))
    len_aggr = int(fattree_k / 2)
    len_edge = int(fattree_k / 2)
    switch_num = len_core + (len_aggr + len_edge) * fattree_k

    logging.info("【heap】: time interval = {}, fattree_k = {}, x = {}, theta = {} over".format(time_interval, fattree_k, x, theta))

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
    save_data_dir = "univ1_trace/univ1_[{}-{}]_epoch{}_fattree{}/x{}_theta{}_multi_heap{}/".format(file_begin, file_end, time_interval, fattree_k, x, theta, heap_num)
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    # read groundtruth data
    volume_groundtruth = pd.read_csv(switch_data_dir + "groundtruth_volume.csv")

    # res file
    set_res_file()

    # simulate
    simulate(fattree_k=fattree_k)
    logging.info("【multi】: time interval = {}, fattree_k = {}, x = {}, theta = {}, heap_num = {} over".format(time_interval, fattree_k, x, theta, heap_num))








