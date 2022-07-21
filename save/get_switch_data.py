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


class fatTreeTopology:
    packet_in_fly = pd.DataFrame()
    packet_all = pd.DataFrame()

    # network = physicalNetwork()

    def __init__(self, k):
        self.k = k
        self.core = []
        self.pod_set = []

    class pod:

        def __init__(self, k, pod_id):
            self.k = k
            self.pod_id = pod_id
            self.aggr = []
            self.edge = []

        def initializePod(self):
            len_core = int(pow(self.k / 2, 2))
            len_aggr = int(self.k / 2)
            len_edge = int(self.k / 2)

            # initialize aggr
            for i in range(len_aggr):
                switch_id = len_core + self.pod_id * self.k + i
                neighbors = []
                for j in range(len_aggr * i, len_aggr * (i + 1)):
                    neighbors.append(i)

                # topology add switch
                self.aggr.append(switch_id)

                # physical network add switch
                # fatTreeTopology.network.addSwitch(switch_id=switch_id, neighbors=neighbors)

            # initialize edge
            for i in range(len_edge):
                switch_id = len_core + self.pod_id * self.k + len_aggr + i

                # topology add switch
                self.edge.append(switch_id)

                # physical network add switch
                # fatTreeTopology.network.addSwitch(switch_id=switch_id, neighbors=self.aggr)

        def getAggr(self, i):
            return self.aggr[i]

        def getEdge(self, i):
            return self.edge[i]

    # initialize both FatTree topology and physical network
    def initializeFatTree(self):

        # initialize core switches
        len_core = int(pow(self.k / 2, 2))
        for i in range(len_core):
            neighbors = []

            # topology add switch
            self.core.append(i)

            # # physical network add switch
            # self.network.addSwitch(switch_id=i, neighbors=neighbors)

        # initialize pods switches
        len_pod_set = self.k
        for i in range(len_pod_set):
            new_pod = self.pod(k=self.k, pod_id=i)
            new_pod.initializePod()
            self.pod_set.append(new_pod)

    # show FatTree topology
    def showFatTree(self):
        logging.info("############################ FatTree {} ################################".format(self.k))

        # core
        logging.info(self.core)

        # pods
        for i in range(len(self.pod_set)):
            pod = self.pod_set[i]
            logging.info("pod id {}".format(pod.pod_id))
            logging.info("aggr {}".format(pod.aggr))
            logging.info("edge {}".format(pod.edge))

    # choose path for a packet, return the path
    def chooseFlowPath(self, flow_id):

        # to return
        packet_path = []

        # generate seed
        random.seed(flow_id)

        # choose pods
        pod_h1 = random.randint(0, self.k - 1)
        pod_h2 = random.randint(0, self.k - 1)

        # choose aggr
        aggr_h1 = random.randint(0, int(self.k / 2) - 1)
        aggr_h2 = random.randint(0, int(self.k / 2) - 1)

        # choose edge
        edge_h1 = random.randint(0, int(self.k / 2) - 1)
        edge_h2 = random.randint(0, int(self.k / 2) - 1)

        # communicate in one pod
        if pod_h1 == pod_h2:

            # [1 switch ] communicate through one edge switch
            if edge_h1 == edge_h2:
                packet_path.append(self.pod_set[pod_h1].getEdge(edge_h1))

            # [3 switch ] communicate through two edge switch, aggr switch
            else:
                packet_path.append(self.pod_set[pod_h1].getEdge(edge_h1))
                packet_path.append(self.pod_set[pod_h1].getAggr(aggr_h1))
                packet_path.append(self.pod_set[pod_h1].getEdge(edge_h2))

        # [5 switch] communicate through two edge switch, two aggr switch, one core switch
        else:
            packet_path.append(self.pod_set[pod_h1].getEdge(edge_h1))
            packet_path.append(self.pod_set[pod_h1].getAggr(aggr_h1))
            packet_path.append(self.core[aggr_h1 * int((self.k / 2)) + aggr_h2])
            packet_path.append(self.pod_set[pod_h2].getAggr(aggr_h1))
            packet_path.append(self.pod_set[pod_h2].getEdge(edge_h2))

        return packet_path

    def addFlow(self, switch_id, flow_id):
        print("a")


def get_switch_data(fattree_k, all_packets):
    global epoch_num, switch_num, time_interval
    topology = fatTreeTopology(fattree_k)
    topology.initializeFatTree()
    topology.showFatTree()

    non_duplicate_packets = all_packets.drop_duplicates(subset=['id', "seq"], keep='first', inplace=False)
    logging.info("raw_packet:{}, non_duplicate_raw_packets:{}".format(len(all_packets), len(non_duplicate_packets)))
    grouped_pcap = all_packets.groupby('epoch')

    for group in grouped_pcap:
        epoch_num = group[0]
        epoch_packet = group[1]
        non_duplicate_epoch_packets = epoch_packet.drop_duplicates(subset = ['id','seq'], keep='first', inplace=False)
        logging.info("############################ 【epoch {}】{} packets, non_duplicate {} packets, time interval = {} ################################".format(epoch_num, len(epoch_packet), len(non_duplicate_epoch_packets), time_interval))
        
        # if epoch_num == 5:
        #     break
        
        # get flow to its route
        switch_counter = [0] * switch_num

        for index, row in non_duplicate_epoch_packets.iterrows():

            # get packet path
            flow_id = row["id"]        
            packet_path = topology.chooseFlowPath(flow_id=flow_id)

            # add flow to switch
            for switch_id in packet_path:
                switch_counter[switch_id] += 1

                switch_file_dir = save_dir + "switch_data/switch{}.csv".format(switch_id)

                # ["switch_id", "epoch", "id", "seq", "timestamp", "new_id", "hash"]
                packet_data = [[switch_id, epoch_num, row['id'], row['seq'], row['timestamp'], row['new_id'], row['hash']]]

                with open(switch_file_dir, mode="a", newline="") as switch_file:
                    writer = csv.writer(switch_file)
                    writer.writerows(packet_data)

        # some epoch log for every switch
        for switch_id in range(0, switch_num):

            # logging.info("【epoch {}】switch{}: has {} packets".format(epoch_num, switch_id, switch_counter[switch_id]))

            switch_count_file = save_dir + "switch_counter.csv"
            switch_data = [[epoch_num, switch_id, switch_counter[switch_id]]]

            with open(switch_count_file, mode="a", newline="") as switch_file:
                writer = csv.writer(switch_file)
                writer.writerows(switch_data)

def set_res_file():
    """
    switch packets_through track file
    """ 
    for switch_id in range(0, switch_num):
        if not os.path.exists(save_dir + "switch_data/"):
            os.makedirs(save_dir + "switch_data/")
        switch_file_dir = save_dir + "switch_data/switch{}.csv".format(switch_id)
        header = ["switch_id", "epoch", "id", "seq", "timestamp", "new_id", "hash"]

        if os.path.exists(switch_file_dir):
            os.remove(switch_file_dir)
        os.mknod(switch_file_dir)

        logging.info("{} set already! ".format(switch_file_dir))

        with open(switch_file_dir, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)

    """
    switch counter track file
    """
    switch_count_file = save_dir + "switch_counter.csv"
    header = ['epoch', 'switch_id', 'counter']

    if os.path.exists(switch_count_file):
        os.remove(switch_count_file)
    os.mknod(switch_count_file)

    logging.info("{} set already! ".format(switch_count_file))

    with open(switch_count_file, mode="a", newline="") as counter_file:
        writer = csv.writer(counter_file)
        writer.writerow(header)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    global epoch_num, theta, alpha, sample_num, file_begin, file_end, time_interval, switch_maxhash_per_epoch
    switch_maxhash_per_epoch = pd.DataFrame()

    # dataset = sys.argv[1]       
    # file_begin = int(sys.argv[2])
    # file_end = int(sys.argv[3])
    # time_interval = int(sys.argv[4])
    # fattree_k = int(sys.argv[5])

    dataset = 'univ1'
    file_begin = 1
    file_end = 20
    time_interval = 95
    fattree_k = 4

    len_core = int(pow(fattree_k / 2, 2))
    len_aggr = int(fattree_k / 2)
    len_edge = int(fattree_k / 2)

    switch_num = len_core + (len_aggr + len_edge) * fattree_k

    pcap_file = pd.DataFrame(pd.read_csv("{}_trace/data/{}_[{}-{}]_epoch{}.csv".format(dataset, dataset,file_begin, file_end, time_interval)))
    pcap_file = pcap_file[['id', 'seq', 'timestamp', 'epoch', 'new_id', 'hash']]

    # save dir
    save_dir = "{}_trace/{}_[{}-{}]_epoch{}_fattree{}/".format(dataset, dataset, file_begin, file_end, time_interval, fattree_k)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    set_res_file()
    """
    switch get data
    """
    get_switch_data(fattree_k, pcap_file)






