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
from tensorflow.keras.models import Sequential, load_model

class programmableSwitch:

    def __init__(self, switch_id):
        self.switch_id = switch_id
        self.p_threshold = 0.3
        self.should_max_hash_track = []
        self.packets_through = pd.DataFrame()
        self.packets_sampled = pd.DataFrame()
        self.sampled_max_hash = 0.0
    
    def initializeSwitch(self, epoch_i):
        global switch_data_dict
        self.packets_through = switch_data_dict[self.switch_id].get_group(epoch_i)
        # logging.info("switch{}, self.packets_through.columns={}".format(self.switch_id, self.packets_through.columns))
    
    def sample_packets(self):
        self.packets_sampled = self.packets_through[self.packets_through['hash'] < self.p_threshold]
        # self.packets_sampled.info()

    def adjust_self_sampling_rate(self):
        global x, alpha, hash_file_name, epoch_i, look_back

        # actual switch max hash
        self.sampled_max_hash = self.packets_sampled['hash'].max()

        # should switch max hash
        self.packets_through =  self.packets_through.reset_index(drop=True)
        self.packets_through.sort_values(by="hash", inplace=True, ascending=True)

        # estimated packets num
        should_max_hash = (x-1) / len(self.packets_through)
        self.should_max_hash_track.append([should_max_hash])

        if epoch_i < look_back - 1:

            # now epoch
            now_p_threshold = self.p_threshold
            self.p_threshold = (1 - alpha) * self.p_threshold + alpha * (x-1) / len(self.packets_through) #     (x / ((actual_k - 1) / self.sampled_max_hash)) 1000/(800/HASH)
           
            # next epoch
            next_p_threshold = self.p_threshold

            logging.info("【TASK3】switch {}: old_p_threshold={} --> adjust to new_p_threshold={}".format(self.switch_id, now_p_threshold, next_p_threshold))
            
            # write hash track file
            hash_data = [[epoch_i, switch_id, len(self.packets_through), len(self.packets_sampled), x, self.sampled_max_hash, should_max_hash, should_max_hash-self.sampled_max_hash, next_p_threshold]]
            with open(hash_file_name, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(hash_data)
        
        else:
            self.should_max_hash_track.pop(0)
        
        # self.packets_through.to_csv(save_data_dir + "[hash pic]/switch{}/epoch{}_track.csv".format(self.switch_id, epoch_i))


class physicalNetwork:

    def __init__(self, fattree_k):
        self.fattree_k = fattree_k
        self.networkSwitches = {}

        self.all_sampled_packet = pd.DataFrame()
        self.effective_x = 0  
        self.effective_x_packet = pd.DataFrame()   

        self.estimated_volume = 0
        self.global_sampling_rate = 0.0

    def initializeNetwork(self, epoch_i):
        global switch_num
        logging.info("###################################### epoch {} ######################################".format(epoch_i))
        for switch_id in range(switch_num):
            # self.networkSwitches[switch_id] = programmableSwitch(switch_id, epoch_i)
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

        # top x hash
        self.all_sampled_packet.sort_values(by="hash", inplace=True, ascending=True)
        self.all_sampled_packet.drop_duplicates(subset=['hash'], keep='first', inplace=True)
        self.effective_x_packet = self.all_sampled_packet.head(self.effective_x)

    def estimate_volume(self):
        global V_file_name, epoch_i, epoch_i
        self.all_sampled_packet =  self.all_sampled_packet.reset_index(drop=True)
        should_x_hash = self.all_sampled_packet.at[x-1, 'hash']

        h = self.effective_x_packet['hash'].max()
        k = len(self.effective_x_packet)

        self.estimated_volume = int((k - 1) / h)
        actual_volume = volume_groundtruth.at[epoch_i, 'actual_volume']
        error = (int(self.estimated_volume) - int(actual_volume)) / actual_volume
        logging.info("【TASK1】VOLUME ESTIMATION = {}, actual volume = {}, ERROR = {}; effectice_sampled_packets_num = {}, max_hash = {}, should_max_hash={}".format(self.estimated_volume, actual_volume, 100 * error, k, h, should_x_hash))

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

    def adjust_switch_sampling_rate(self, model):
        global epoch_i, look_back

        testX = []
        for switch_id in range(switch_num):
            self.networkSwitches[switch_id].adjust_self_sampling_rate()
            if epoch_i >= look_back - 1:
                testX.append(self.networkSwitches[switch_id].should_max_hash_track)

        if epoch_i >= look_back - 1:
            testX = np.concatenate(testX, axis=0)
            testX = np.array(testX)
            testX = np.reshape(testX, (testX.shape[0], testX.shape[1] , 1)) 
            testPredict = model.predict(testX)

            for switch_id in range(switch_num):
                # len_packets_through
                len_packets_through = len(self.networkSwitches[switch_id].packets_through)

                # p_threshold
                now_p_threshold = self.networkSwitches[switch_id].p_threshold
                self.networkSwitches[switch_id].p_threshold = testPredict[switch_id][0]
                next_p_threshold = self.networkSwitches[switch_id].p_threshold
                logging.info("【TASK3】switch {}: old_p_threshold={} --> adjust to new_p_threshold={}".format(switch_id, now_p_threshold, next_p_threshold))

                # should_max_hash
                if len(self.networkSwitches[switch_id].packets_through) >= x:
                    should_max_hash = self.networkSwitches[switch_id].packets_through.hash.nsmallest(x).iloc[-1] # self.packets_through.at[sample_num-1, 'hash'] # 改了这里变成sample_num - 1
                else:
                    should_max_hash = -1

                # len_packets_sampled
                len_packets_sampled = len(self.networkSwitches[switch_id].packets_sampled)

                # sample_max_hash
                sample_max_hash = self.networkSwitches[switch_id].sampled_max_hash

                # write hash track file
                hash_data = [[epoch_i, switch_id, len_packets_through, len_packets_sampled, x, sample_max_hash, should_max_hash, should_max_hash-sample_max_hash, next_p_threshold]]
                with open(hash_file_name, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(hash_data)

    def reset_network(self):
        # reset switches
        for switch_id in self.networkSwitches.keys():
            self.networkSwitches[switch_id].packets_through = pd.DataFrame()
            self.networkSwitches[switch_id].packets_sampled = pd.DataFrame()
            self.networkSwitches[switch_id].sampled_max_hash = 0.0

        # reset central controller
        self.all_sampled_packet = pd.DataFrame()
        self.effective_x = 0
        self.effective_x_packet = pd.DataFrame()

        self.estimated_volume = 0
        self.global_sampling_rate = 0.0


"""
simulate
"""
def simulate(fattree_k):

    global switch_num, epoch_len, epoch_i

    network = physicalNetwork(fattree_k)
    for switch_id in range(switch_num):
        network.networkSwitches[switch_id] = programmableSwitch(switch_id)

    model = load_model("univ1_trace/univ1_[{}-{}]_epoch-{}_fattree{}_x{}.h5".format(file_begin, file_end, time_interval, fattree_k, 1000))
    logging.info("{} epoches in all".format(epoch_len+1))
    for epoch_i in range(epoch_len+1):
        network.initializeNetwork(epoch_i)
        network.merge_data()
        network.estimate_volume()
        network.check_heavy_hitter()
        network.adjust_switch_sampling_rate(model)
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

    file_begin = 1
    file_end = 20
    time_interval = 30
    fattree_k = 4

    x = 2000
    theta = 0.01
    alpha = 0.8
    look_back = 10

    len_core = int(pow(fattree_k / 2, 2))
    len_aggr = int(fattree_k / 2)
    len_edge = int(fattree_k / 2)
    switch_num = len_core + (len_aggr + len_edge) * fattree_k

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
    save_data_dir = "univ1_trace/univ1_[{}-{}]_epoch{}_fattree{}/x{}_theta{}/".format(file_begin, file_end, time_interval, fattree_k, x, theta)
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    # read groundtruth data
    volume_groundtruth = pd.read_csv(switch_data_dir + "groundtruth_volume.csv")

    # res file
    set_res_file()

    # simulate
    simulate(fattree_k=fattree_k)

    logging.info("new: time interval = {}, fattree_k = {}, x = {}, theta = {} over".format(time_interval, fattree_k, x, theta))














