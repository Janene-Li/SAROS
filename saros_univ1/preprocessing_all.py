
import sys
from scapy.all import *
import pandas as pd
import csv
from datetime import datetime
import logging
import os
import random

LIMIT = 100000000
count = 0
pcap_dataframe = pd.DataFrame(columns=["id", "seq", "timestamp"])
tick = []


def parse_packets(pkt):
    global LIMIT, count, pcap_dataframe
    try:
        if str(pkt[1].proto) == '6':
            id = str(pkt[1].src) + "." + str(pkt[1].dst) + "." + str(pkt[1].sport) + "." + str(pkt[1].dport) + "." + str(pkt[1].proto)
            seq = pkt[1].seq
            timestamp = datetime.fromtimestamp(int(pkt.time))
            pkt_dataframe = pd.DataFrame([[id, seq, timestamp]], columns=["id", "seq", "timestamp"])
            if pcap_dataframe.empty:
                pcap_dataframe = pkt_dataframe
            else:
                pcap_dataframe = pd.concat([pcap_dataframe, pkt_dataframe], axis=0)

        count = count + 1
        if count % 1000 == 0:
            logging.info("processing {} packets".format(count))

    except AttributeError:
        return False

    if 0 < LIMIT <= count:
        return True
    else:
        return False


def clock(timestamp, time_interval, epoch_num):
    global tick, epoch_packet_num
    if timestamp not in tick:  # 已经有time_interval不同时间戳，来的新数据包时间戳不在里面，就更新
        epoch_packet_num = 0
        if len(tick) + 1 > int(time_interval):
            logging.info("############################ Epoch {} End (time_interval={})################################".format(epoch_num, time_interval))
            epoch_num += 1
            tick = []
            tick = [str(timestamp)]
        else:
            tick = tick + [str(timestamp)]
            # logging.info(len(tick))
    else:
        epoch_packet_num += 1
        tick = tick
    # logging.info(epoch_packet_num)
    return epoch_num


def get_epoch(packet, time_interval):
    global epoch_packet_num

    packet['epoch'] = ''

    packet['seq'] = packet['seq'].apply(str)
    packet['new_id'] = packet.apply(lambda x:x['id'] + "." + x['seq'], axis=1)
    # packet['new_id'] = packet['id'] + "." + packet['seq']

    packet['hash'] = ''

    epoch_packet_num = 0
    epoch_num = 0

    for index, row in packet.iterrows():

        timestamp = row['timestamp']
        epoch_num = clock(timestamp, time_interval, epoch_num)
        packet.at[index, 'epoch'] = epoch_num

        random.seed(row['new_id'])
        hash_value = random.uniform(0,1)
        packet.at[index, 'hash'] = hash_value

    return packet


def divide_epoch(packet, time_interval):
    packet['epoch'] = 0
    group_by_timestamp = packet.groupby('timestamp')

    clock = 0

    epoch_file = "univ1_trace/univ1_[{}-{}]_epoch{}.csv".format(file_begin, file_end, epoch_interval)
    if os.path.exists(epoch_file):
        os.remove(epoch_file)
    os.mknod(epoch_file)

    header = ["id", "seq", "timetsamp", "epoch"]
    with open(epoch_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(header)

    for group in group_by_timestamp:

        clock += 1        
        epoch = clock // int(time_interval)

        logging.info("group index={}, epoch= {}, time interval={}".format(clock, epoch, time_interval))
        group[1]['epoch'] = epoch

        pd.DataFrame(group[1]).to_csv(epoch_file, mode='a', header=True)
    # group_by_timestamp = pd.DataFrame(group_by_timestamp)

    return group_by_timestamp



if "__main__" == __name__:

    logging.getLogger().setLevel(logging.INFO)

    path='univ1_trace/parse/'
    file_begin = 1
    file_end = 20
    epoch_interval = 50

    """
    # 1. parse packets with sniff
    """
    for i in range(file_begin, file_end + 1):

        pcap_name = "univ1_pt{}".format(i)
        pcap_file = pcap_name + ".pcap"
        pcap_file_dir = path + pcap_file

        if os.path.exists (pcap_file_dir+".csv"):
            continue
        
        logging.info("parse pcap_file: {} #############################################".format(pcap_file_dir))
        sniff(offline=pcap_file_dir, stop_filter=parse_packets, store=False)

        logging.info("pcap_file {} over, {} packets in all ######################".format(pcap_file, count))
        pcap_dataframe.to_csv(pcap_file_dir + ".csv")
        pcap_dataframe = pd.DataFrame(columns=["id", "seq", "timestamp"])

    """
    # 2. combile all the .pcap.csv file
    """
    combiled_df = pd.DataFrame()

    for i in range(file_begin, file_end + 1):

        csv_name = "univ1_pt{}".format(i)
        csv_file = csv_name + ".pcap.csv"
        csv_file_dir = path + csv_file

        logging.info("combine csv_file: {} #############################################".format(csv_file_dir))
        now_df = pd.DataFrame(pd.read_csv(csv_file_dir, index_col=None))
        combiled_df = combiled_df.append(now_df[["id", "seq", "timestamp"]])

    combiled_df =  combiled_df.reset_index(drop=True)   # 要处理好索引，不然处理不对
    combiled_df.info()
    logging.info(combiled_df.index)

    """
    # 3. data get epoch
    """

    logging.info("get epoch with interval = {} #############################################".format(epoch_interval))
    # pcap_dataframe = pd.read_csv("univ1_trace/univ1_pt1.csv")
    # pcap_dataframe = divide_epoch(packet=combiled_df, time_interval=epoch_interval)
    pcap_dataframe = get_epoch(packet=combiled_df, time_interval=epoch_interval)
    pcap_dataframe.info()
    pcap_dataframe.to_csv("univ1_trace/univ1_[{}-{}]_epoch{}.csv".format(file_begin, file_end, epoch_interval))
