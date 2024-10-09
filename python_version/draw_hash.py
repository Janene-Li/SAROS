# 导入我们所需的库 as：即给库取别名，方便书写
from genericpath import exists
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import logging
import pandas as pd
import os

def get_raw_data(dir):    

    raw_df = pd.DataFrame(pd.read_csv(dir))
    raw_df = raw_df[(raw_df.should_max_hash != -1)]
    raw_df.info()
    grouped_df = raw_df.groupby('switch_id')

    for switch_df in grouped_df:

        switch_id = switch_df[0]
        switch_hash = switch_df[1]

        switch_hash['should_diff'] = switch_hash['should_max_hash'].diff()
        logging.info("【switch {}】{} epoches ###################\n".format(switch_id, len(switch_hash)))

        # draw_switch_hash(switch_hash['epoch'], switch_hash['should_max_hash'], switch_hash['adjust_max_hash'].shift(axis=0), switch_id)
        # draw_switch_hash_diff(switch_hash['epoch'], switch_hash['should_diff'], switch_id)
        draw_num_hash(switch_hash['len_packets_through'], switch_hash['should_max_hash'], switch_id)


def draw_switch_hash(x_axis_data, y1_axis_data, y2_axis_data, switch_id):

    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
    plt.figure(figsize=(80, 5))  # figsize:确定画布大小 
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x_axis_data, y1_axis_data, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='actual')
    plt.plot(x_axis_data, y2_axis_data, 'ro-', color='green', alpha=0.8, linewidth=1, label='predict')

    # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    plt.legend(loc="upper right")
    plt.xlabel('epoch')
    plt.ylabel('hash')

    save_dir = 'univ1_trace/univ1_[{}-{}]_epoch{}_theta{}_x{}/[hash track]/'.format(file_begin, file_end, time_interval, theta, sample_num)
    plt.savefig(save_dir + 'switch{}.jpg'.format(switch_id))


def draw_switch_hash_diff(x, y, switch_id):

    # 绘图
    # 1. 确定画布
    plt.figure(figsize=(60, 5))  # figsize:确定画布大小 

    # 2. 绘图
    plt.scatter(x,  # 横坐标
                y,  # 纵坐标
                
                c='red',  # 点的颜色
                linewidth=1,
                label='should hash diff')  # 标签 即为点代表的意思
    # 3.展示图形
    # plt.legend()  # 显示图例
    # plt.show()  # 显示所绘图形
    plt.xlabel('epoch')
    plt.ylabel('hash_diff')

    # 4.保存图形
    save_dir = 'univ1_trace/univ1_[{}-{}]_epoch{}_x{}_theta{}/[hash track]/'.format(file_begin, file_end, time_interval, sample_num, theta)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(save_dir + 'switch{}.jpg'.format(switch_id))

def draw_num_hash(x, y, switch_id):
    # 定义数据
    # x = np.random.rand(10)  # 取出10个随机数
    # y = x + x ** 2 - 10  # 用自定义关系确定y的值

    # 绘图
    # 1. 确定画布
    plt.figure(figsize=(10, 5))  # figsize:确定画布大小 

    # 2. 绘图
    plt.scatter(x,  # 横坐标
                y,  # 纵坐标
                c='#4169E1',  # 点的颜色
                linewidth=1,
                label='num-hash')  # 标签 即为点代表的意思
    # 3.展示图形
    # plt.legend()  # 显示图例
    # plt.show()  # 显示所绘图形
    plt.legend(loc="upper right")
    plt.xlabel('packets num')
    plt.ylabel('hash')
    plt.xlim(0,58000)
    plt.ylim(0,1.1)

    # 4.保存图形
    save_dir = 'univ1_trace/univ1_[{}-{}]_epoch{}_theta{}_x{}/[num-hash]/'.format(file_begin, file_end, time_interval, theta, sample_num)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + 'switch{}.jpg'.format(switch_id))

if "__main__" == __name__:
    logging.getLogger().setLevel(logging.INFO)
    
    sample_num = 1000
    alpha = 0.8
    theta = 0.01
    time_interval = 10
    file_begin = 1
    file_end = 20

    raw_data_dir = "univ1_trace/univ1_[{}-{}]_epoch{}_theta{}_x{}/[hash track].csv".format(file_begin, file_end, time_interval, theta, sample_num)

    get_raw_data(raw_data_dir)



