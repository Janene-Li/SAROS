import pandas as pd
import random
import logging
import sys
import csv
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class programmableSwitch:

    def __init__(self, switch_id):
        self.switch_id = switch_id
        self.p_threshold = 0.3
        self.packets_through = pd.DataFrame()
        self.packets_sampled = pd.DataFrame()
        self.switch_max_hash = 0.0

    """
    # DECIDE WHETHER TO SAMPLE
    """
    def samplePacket(self, packet):
        flow_id = packet['id']
        packet_id = packet['seq']

        # generate hash value
        random.seed(flow_id + "." + str(packet_id))
        hash_value = random.uniform(0,1)
        packet['hash'] = hash_value
        packet['new_id'] = flow_id + "." + str(packet_id)

        # add to packet through
        self.packets_through = self.packets_through.append(packet)

        # add to packet sampled
        if hash_value < self.p_threshold:
            self.packets_sampled = self.packets_sampled.append(packet)

    """
    # SWITCH ITSELF ADJUST SAMPLING RATE
    """
    def selfAdjustSamplingRate(self):
        global sample_num, alpha, switch_maxhash_per_epoch

        # actual switch max hash
        self.switch_max_hash = self.packets_sampled['hash'].max()

        # should switch max hash
        self.packets_through =  self.packets_through.reset_index(drop=True)
        if len(self.packets_through) >= sample_num:
            should_max_hash = self.packets_through.at[sample_num-1, 'hash'] # df.at[4, 'B']
        else:
            should_max_hash = -1

        # actual k
        actual_k = len(self.packets_sampled)

        # estimated packets num
        self.p_threshold = (1 - alpha) * self.p_threshold + alpha * (sample_num / ((actual_k - 1) / self.switch_max_hash))
        
        # file_name
        file_name = "univ1_trace/univ1_[{}-{}]_epoch{}_[hash track]_x{}.csv".format(file_begin, file_end, time_interval, sample_num)
        # data: ["epoch","switch_id", "actual_sample_num", "should_sample_num", "actual_max_hash", "should_max_hash", "adjust_max_hash", "should-actual", "should-adjust"]
        row_data = [[epoch_num, self.switch_id, actual_k, sample_num, self.switch_max_hash, should_max_hash, self.p_threshold, should_max_hash-self.switch_max_hash, should_max_hash-self.p_threshold]]

        # write hash track file
        with open(file_name, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(row_data)

        # switch_maxhash_per_epoch = switch_maxhash_per_epoch.append([{"epoch": epoch_num,
        #                                                              "switch_id": self.switch_id,
        #                                                              "actual_sample_num": k,
        #                                                              "should_sample_num": sample_num,
        #                                                              "actual_max_hash": self.switch_max_hash,
        #                                                              "should_max_hash": should_max_hash,
        #                                                              "adjust_max_hash": self.p_threshold
        #                                                            }], ignore_index=True)


    """
    GIVE SOME MESSAGE WHEN REPORT AT THE END OF EPOCH
    """
    def sampleInfo(self):
        # logging.info("【TASK0】switch {}: {}\n{}".format(self.switch_id, self.packets_sampled.columns, self.packets_sampled.head(2)))
        epoch_sampling_rate = len(self.packets_sampled) / len(self.packets_through)

        # sample_packet_heap = pd.DataFrame()
        # sample_packet_heap = self.packets_sampled.sort_values(by="hash",ascending=True, inplace=False)
        # sample_max_hash = sample_packet_heap.iloc[-1].tolist()
        # sample_max_hash = sample_max_hash[4]
        sample_max_hash = self.packets_sampled['hash'].max()

        logging.info("【TASK0】switch {}: sample {} packets, sampling_rate={}, sample_max_hash={}".format(self.switch_id, len(self.packets_sampled), epoch_sampling_rate, sample_max_hash))
    

class centralController:

    def __init__(self):
        self.HH_set = pd.DataFrame()
        self.all_sampled_packet = pd.DataFrame()
        self.effective_sampled_packet = pd.DataFrame()
        self.effective_x = 0
        self.global_sampling_rate = 0.0

    def volumeEstimate(self):
        # h = self.effective_sampled_packet[-1:]['hash']
        # h = self.effective_sampled_packet.iloc[-1].tolist()
        # h = h[-1]    # 这里可能会出错，不过不知道为什么列的顺序会变
        h = self.effective_sampled_packet['hash'].max()
        k = len(self.effective_sampled_packet)
        estimated_volume = int((k - 1) / h)
        logging.info("【TASK1】VOLUME ESTIMATION = {}, effectice_sampled_packets_num = {}, max_hash = {}".format(estimated_volume, k, h))
        return estimated_volume

    def checkHeavyHitter(self):
        global epoch_num, theta

        # calculate global sampling rate
        estimated_volume = self.volumeEstimate()
        self.effective_x = len(self.effective_sampled_packet)
        self.global_sampling_rate = self.effective_x/estimated_volume
        global_threshold = int(estimated_volume * theta) * self.global_sampling_rate

        CM_sketch = {}

        # CM sketch take a count
        for index, row in self.effective_sampled_packet.iterrows():
            flow_id = row['id']
            if flow_id in CM_sketch.keys():
                CM_sketch[flow_id] += 1
            else:
                CM_sketch[flow_id] = 1

        # check heavy hitter
        for flow_id, counter in CM_sketch.items():
            if CM_sketch[flow_id] > global_threshold:
                self.HH_set = self.HH_set.append([{"epoch": epoch_num,
                                                   "id": flow_id,
                                                   "effective_x": self.effective_x, 
                                                   "sampling_rate": self.global_sampling_rate,
                                                   "global_threshold": int(estimated_volume * theta), # global_threshold,
                                                   "packets_num": counter,
                                                   "estimated_packets_num": counter / self.global_sampling_rate
                                                   }
                                                  ], ignore_index=True)
        logging.info("【TASK2】HH DETECTION: global_sampling_rate = {}, theta = {}, global_threshold = {} ".format(self.global_sampling_rate, theta, global_threshold))


class physicalNetwork:
    def __init__(self, k):
        self.k = k
        self.networkSwitches = {}
        self.controller = centralController()
        self.estimate_volume = 0
        self.effective_x = 0

    def initializeSwitch(self):
        len_core = int(pow(self.k / 2, 2))
        len_aggr = int(self.k / 2)
        len_edge = int(self.k / 2)
        switch_num = len_core + len_aggr * self.k + len_edge * self.k
        for i in range(switch_num):
            self.networkSwitches[i] = programmableSwitch(i)
        logging.info("switch num in FatTree(k={}) is {}".format(self.k, switch_num))

    """
    1. programmable switch will work
    """
    def processPacket(self, packet, switch_id_list):
        for switch_id in switch_id_list:
            # now_switch = self.networkSwitches[switch_id]
            # now_switch.samplePacket(packet)
            self.networkSwitches[switch_id].samplePacket(packet)

    """
     2 communicate between switches and controller
    """
    def mergeReport(self):

        # decide how many packet to summary

        all_sampled_packet = pd.DataFrame()
        all_sampled_packet['switch_id'] = ''

        for switch_id in self.networkSwitches.keys():

            switch = self.networkSwitches[switch_id]
            packets_switch_sampled = switch.packets_sampled
            packets_switch_sampled['switch_id'] = switch_id

            # logging.info("switch {}: sample {} packets".format(switch_id, len(packets_switch_sampled)))

            if self.effective_x == 0:
                self.effective_x = len(packets_switch_sampled)
            else:
                self.effective_x = min(self.effective_x, len(packets_switch_sampled))

            all_sampled_packet = all_sampled_packet.append(packets_switch_sampled)

        for switch_id in self.networkSwitches.keys():
            switch = self.networkSwitches[switch_id]
            
        logging.info("effective sampled packets number is {}".format(self.effective_x))

        # merge all the packets received
        before_merge = len(all_sampled_packet)
        all_sampled_packet = pd.DataFrame(all_sampled_packet)
        all_sampled_packet.drop_duplicates(subset=['new_id'], keep='first', inplace=True)
        after_merge = len(all_sampled_packet)
        logging.info("Merge {} packets to {} packets".format(before_merge, after_merge))
        # logging.info("effective packets:{}".format(all_sampled_packet.head(2)))

        # get effective packets
        all_sampled_packet.sort_values(by="hash", inplace=True, ascending=True)
        self.controller.all_sampled_packet = all_sampled_packet
        all_sampled_packet.drop_duplicates(subset=['hash'], keep='first', inplace=True)
        self.controller.effective_sampled_packet = all_sampled_packet.head(self.effective_x)

    """
     3 central controller will work
    """
    def processReport(self):
        self.estimate_volume = self.controller.volumeEstimate()
        self.controller.checkHeavyHitter()

    def adjustSamplingRate(self):
        global alpha, sample_num

        for switch_id in self.networkSwitches.keys():

            old_p_threshold = self.networkSwitches[switch_id].p_threshold

            # new_sample_max_hash = (1 - alpha) * switch.p_threshold + alpha * (switch.p_threshold + (sample_num - len(switch.packets_sampled))/self.estimate_volume)
            self.networkSwitches[switch_id].selfAdjustSamplingRate()

            new_p_threshold = self.networkSwitches[switch_id].p_threshold

            logging.info("【TASK3】switch {}: old_p_threshold={} ----adjust to----> new_p_threshold={}".format(switch_id, old_p_threshold, new_p_threshold))

    def resetNetwork(self):

        # reset switches
        for switch_id in self.networkSwitches.keys():
            self.networkSwitches[switch_id].packets_through = pd.DataFrame()
            self.networkSwitches[switch_id].packets_sampled = pd.DataFrame()
            self.networkSwitches[switch_id].switch_max_hash = 0.0

        # reset central controller
        self.controller.all_sampled_packet = pd.DataFrame()
        self.controller.effective_sampled_packet = pd.DataFrame()
        self.controller.effective_x = 0
        self.controller.global_sampling_rate = 0.0

        # reset physical network
        self.estimate_volume = 0
        self.effective_x = 0


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


def simulate(fattree_k, raw_pcap):
    global epoch_num
    topology = fatTreeTopology(fattree_k)
    topology.initializeFatTree()
    topology.showFatTree()

    network = physicalNetwork(fattree_k)
    network.initializeSwitch()

    non_duplicate_packets = raw_pcap.drop_duplicates(subset=['id', "seq"], keep='first', inplace=False)
    logging.info("raw_packet:{}, non_duplicate_raw_packets:{}".format(len(raw_pcap), len(non_duplicate_packets)))
    # grouped_pcap = raw_pcap.groupby('epoch')
    grouped_pcap = raw_pcap.groupby('epoch')

    volume_estimate_result = pd.DataFrame()

    for group in grouped_pcap:
        epoch_num = group[0]
        epoch_packet = group[1]
        non_duplicate_epoch_packets = epoch_packet.drop_duplicates(subset = ['id','seq'], keep='first', inplace=False)
        logging.info("############################ 【epoch {}】{} packets, non_duplicate {} packets ################################".format(epoch_num, len(epoch_packet), len(non_duplicate_epoch_packets)))
        
        # if epoch_num == 1:
        #     break

        """
        # SWITCH PART
        # FOR EACH PACKET
        # 1. GET PACKET PATH
        # 2. ADD PACKET TO SWICTHES IN PATH
        """

        logging.info("【switch part】")
        count = 0
        for index, row in non_duplicate_epoch_packets.iterrows():
            count += 1
            if count % 1000 == 0:
                logging.info("{} packets fly in network".format(count))
            flow_id = row["id"]

            # get packet path
            packet_path = topology.chooseFlowPath(flow_id=flow_id)

            # add packet to switches in path
            network.processPacket(row, packet_path)


        """
        # CONTROLLER PART
        # FOR EACH EPOCH
        # 1. CONTROLLER WILL GET SAMPLED PACKETS FROM EVERY SWITCH
        # 2. CONTROLLER WILL CHECK HEAVY HITTER
        # 3. CONTROLLER WILL UPDATE p_threshold IN EVERY SWITCH
        """
        logging.info("【controller part】")
        network.mergeReport()
        # network.processReport()
        network.estimate_volume = network.controller.volumeEstimate()
        network.controller.checkHeavyHitter()
        network.adjustSamplingRate()

        error = (network.estimate_volume - len(non_duplicate_epoch_packets)) / len(non_duplicate_epoch_packets)
        logging.info("【ERROR】{}%".format(100 * error))
        volume_estimate_result = volume_estimate_result.append([{"epoch": epoch_num,
                                                                 "actual_volume": len(non_duplicate_epoch_packets),
                                                                 "estimated_volume": network.estimate_volume,
                                                                 "error%": 100 * error
                                                                }], ignore_index=True)
        """
        # RESET CONTROLLER
        # RESET SWITCHES
        """
        network.resetNetwork()

    sum_actual_volume = volume_estimate_result['actual_volume'].sum()
    sum_estimate_volume = volume_estimate_result['estimated_volume'].sum()
    sum_error = (sum_estimate_volume - sum_actual_volume) / sum_actual_volume
    volume_estimate_result = volume_estimate_result.append([{"epoch": epoch_num,
                                    "actual_volume": sum_actual_volume,
                                    "estimated_volume": sum_estimate_volume,
                                    "error%": 100 * sum_error
                                    }], ignore_index=True)

    volume_estimate_result.to_csv("univ1_trace/univ1_[{}-{}]_epoch{}_[volume estimate]_x{}.csv".format(file_begin, file_end, time_interval, sample_num))

    network.controller.HH_set.to_csv("univ1_trace/univ1_[{}-{}]_epoch{}_[heavy hitter]_x{}_theta{}.csv".format(file_begin, file_end, time_interval, sample_num, theta))

    # switch_maxhash_per_epoch.to_csv("univ1_trace/univ1_[{}-{}]_epoch{}_[hash track]_x{}.csv".format(file_begin, file_end, time_interval, sample_num))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    global epoch_num, theta, alpha, sample_num, file_begin, file_end, time_interval, switch_maxhash_per_epoch
    switch_maxhash_per_epoch = pd.DataFrame()

    sample_num = 1000
    alpha = 0.8
    theta = 0.01
    time_interval = 10  
    fattree_k = 4

    file_begin = 1
    file_end = 20

    pcap_file = pd.DataFrame(pd.read_csv("univ1_trace/univ1_[{}-{}]_epoch{}.csv".format(file_begin, file_end, time_interval)))
    pcap_file = pcap_file[['id', 'seq', 'timestamp', 'epoch']]

    # define hash track file
    hash_file_name = "univ1_trace/univ1_[{}-{}]_epoch{}_[hash track]_x{}.csv".format(file_begin, file_end, time_interval, sample_num)
    header = ["epoch","switch_id", "actual_sample_num", "should_sample_num", "actual_max_hash", "should_max_hash", "adjust_max_hash", "should-actual", "should-adjust"]

    # write hash track file
    os.mknod(hash_file_name)
    logging.info("{} set already! ".format(hash_file_name))
    with open(hash_file_name, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

    simulate(fattree_k=fattree_k, raw_pcap=pcap_file)
