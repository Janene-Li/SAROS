import sys
import os
import logging
import pandas as pd
import csv
import argparse

def preprocessing():
    global dataset, path, file_begin, file_end, time_interval
    os.system("python preprocessing_all.py %s %s %s %s" % (dataset, file_begin, file_end, time_interval))

def get_switch_data():
    os.system("python get_switch_data.py %s %s %s %s %s" % (dataset, file_begin, file_end, time_interval, fattree_k))

def kmv():
    os.system("python merge_switch_data_kmv.py %s %s %s %s %s %s %s %s" % (dataset, file_begin, file_end, time_interval, fattree_k, alpha, theta, x))
    os.system("python Analysing.py %s %s %s %s %s %s %s %s %s" % (dataset, file_begin, file_end, time_interval, fattree_k, alpha, theta, x, algorithm))

def lstm():
    os.system("python merge_switch_data_lstm.py %s %s %s %s %s %s %s %s" % (dataset, file_begin, file_end, time_interval, fattree_k, alpha, theta, x))
    os.system("python Analysing.py %s %s %s %s %s %s %s %s %s" % (dataset, file_begin, file_end, time_interval, fattree_k, alpha, theta, x, algorithm))

def ewma():
    os.system("python merge_switch_data_ewma.py %s %s %s %s %s %s %s " % (dataset, file_begin, file_end, time_interval, fattree_k, theta, x))
    os.system("python Analysing.py %s %s %s %s %s %s %s %s %s" % (dataset, file_begin, file_end, time_interval, fattree_k, alpha, theta, x, algorithm))
    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    global dataset, path, file_begin, file_end, time_interval, fattree_k, algorithm

    dataset_list = ["univ1", "univ2", "mawi"]
    dataset = dataset_list[2]

    path='{}_trace/raw/'.format(dataset)
    file_begin = 1
    file_end = 1
    fattree_k = 4
    alpha = 1
    theta = 0.01
    algorithm = "diff"

    parser = argparse.ArgumentParser(description='kmv + Analysing')
    # parser.add_argument('--x', type=int, default=500, help='x')
    parser.add_argument('--time_interval', type=int, default=5, help='time interval')
    args = parser.parse_args()

    # x = args.x    # 500, 1000, 1500
    time_interval = args.time_interval  # 5, 10, 15, 20, 25, 30, 35, 40, 45

    preprocessing()
    # get_switch_data()
    # kmv()
    # lstm()
    # ewma()






