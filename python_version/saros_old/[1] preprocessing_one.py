
import sys
from scapy.all import *
import pandas as pd
import csv
from datetime import datetime
import logging

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
    global tick
    if timestamp not in tick:  # 已经有time_interval不同时间戳，来的新数据包时间戳不在里面，就更新
        if len(tick) + 1 > int(time_interval):
            print("############################ Epoch {} End ################################".format(epoch_num))
            epoch_num += 1
            tick = []
            tick = [str(timestamp)]
        else:
            tick = tick + [str(timestamp)]
    else:
        tick = tick
    return epoch_num


def get_epoch(packet, time_interval):
    epoch_num = 0
    packet['epoch'] = ''
    for index, row in packet.iterrows():
        timestamp = row['timestamp']
        epoch_num = clock(timestamp, time_interval, epoch_num)
        packet.at[index, 'epoch'] = epoch_num
        # packet['epoch'].iloc[index] = epoch_num
        # packet.loc[index, 'epoch'] = epoch_num

    return packet


if "__main__" == __name__:
    logging.getLogger().setLevel(logging.INFO)

    # parse packets with sniff
    sniff(offline="univ1_trace/univ1_pt1.pcap", stop_filter=parse_packets, store=False)
    logging.info("getting timestamp over, {} packets in all".format(count))
    pcap_dataframe.to_csv("univ1_trace/univ1_pt2.csv")

    # data with epoch
    epoch_interval = 3
    pcap_dataframe = pd.read_csv("univ1_trace/univ1_pt1.csv")
    pcap_dataframe = get_epoch(packet=pcap_dataframe, time_interval=epoch_interval)
    pcap_dataframe.to_csv("univ1_trace/univ1_pt1_epoch{}.csv".format(epoch_interval))
