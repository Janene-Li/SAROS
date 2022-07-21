# 导入我们所需的库 as：即给库取别名，方便书写
from genericpath import exists
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import logging
import pandas as pd
import os
from scipy.optimize import curve_fit
import scipy
from scipy import optimize

def get_raw_data(dir_):    

    raw_df = pd.DataFrame(pd.read_csv(dir_))
    raw_df = raw_df[(raw_df.should_max_hash != -1)]
    raw_df.info()
    
    grouped_df = raw_df.groupby('switch_id')

    for switch_df in grouped_df:

        switch_id = switch_df[0]
        switch_hash = switch_df[1]
        switch_hash.sort_values(by="len_packets_through", inplace=True, ascending=True)

        switch_hash['should_diff'] = switch_hash['should_max_hash'].diff()
        logging.info("【switch {}】{} epoches ###################\n".format(switch_id, len(switch_hash)))

        # draw_switch_hash(switch_hash['epoch'], switch_hash['should_max_hash'], switch_hash['adjust_max_hash'].shift(axis=0), switch_id)
        # draw_switch_hash_diff(switch_hash['epoch'], switch_hash['should_diff'], switch_id)
        # draw_num_hash(switch_hash['len_packets_through'], switch_hash['should_max_hash'], switch_id)
        fit_num_hash_formula(switch_hash['len_packets_through'], switch_hash['should_max_hash'], switch_id)



def draw_switch_hash(x_axis_data, y1_axis_data, y2_axis_data, switch_id):

    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
    plt.figure(figsize=(40, 5))  # figsize:确定画布大小 
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


# 一阶函数方程(直线)
def func_0(x, c):
    return c*1/x

# # 一阶函数方程(直线)
# def func_1(x, a, b):
#     return a*x + b


# # 二阶曲线方程
# def func_2(x, a, b, c):
#     return a * np.power(x, 2) + b * x + c


# # 三阶曲线方程
# def func_3(x, a, b, c, d):
#     return a * np.power(x, 3) + b * np.power(x, 2) + c * x + d


# # 四阶曲线方程
# def func_4(x, a, b, c, d, e):
#     return a * np.power(x, 4) + b * np.power(x, 3) + c * np.power(x, 2) + d * x + e



def fit_num_hash_formula(x_arr, y_arr, switch_id):
    # 拟合参数都放在popt里，popt是个数组，参数顺序即你自定义函数中传入的参数的顺序
    popt0, pcov0 = curve_fit(func_0, x_arr, y_arr)
    c0 = popt0[0]
    # popt1, pcov1 = curve_fit(func_1, x_arr, y_arr)
    # a1 = popt1[0]
    # b1 = popt1[1]
    # popt2, pcov2 = curve_fit(func_2, x_arr, y_arr)
    # a2 = popt2[0]
    # b2 = popt2[1]
    # c2 = popt2[2]
    # popt3, pcov3 = curve_fit(func_3, x_arr, y_arr)
    # a3 = popt3[0]
    # b3 = popt3[1]
    # c3 = popt3[2]
    # d3 = popt3[3]
    # popt4, pcov4 = curve_fit(func_4, x_arr, y_arr)
    # a4 = popt4[0]
    # b4 = popt4[1]
    # c4 = popt4[2]
    # d4 = popt4[3]
    # e4 = popt4[4]
    
    yvals0 = func_0(x_arr, c0)
    print("1/x拟合数据为: ", c0)
    # yvals1 = func_1(x_arr, a1, b1)
    # print("一阶拟合数据为: ", yvals1)
    # yvals2 = func_2(x_arr, a2, b2, c2)
    # print("二阶拟合数据为: ", yvals2)
    # yvals3 = func_3(x_arr, a3, b3, c3, d3)
    # print("三阶拟合数据为: ", yvals3)
    # yvals4 = func_4(x_arr, a4, b4, c4, d4, e4)
    # print("四阶拟合数据为: ", yvals4)

    # rr1 = goodness_of_fit(yvals1, y)
    # print("一阶曲线拟合优度为%.5f" % rr1)
    # rr2 = goodness_of_fit(yvals2, y)
    # print("二阶曲线拟合优度为%.5f" % rr2)
    # rr3 = goodness_of_fit(yvals3, y)
    # print("三阶曲线拟合优度为%.5f" % rr3)
    # rr4 = goodness_of_fit(yvals4, y)
    # print("四阶曲线拟合优度为%.5f" % rr4)

    figure3 = plt.figure(figsize=(8,6))
    plt.plot(x_arr, yvals0, color="green", label='0')
    # plt.plot(x_arr, yvals1, color="#72CD28", label='1')
    # plt.plot(x_arr, yvals2, color="#EBBD43", label='2')
    # plt.plot(x_arr, yvals3, color="#50BFFB", label='3')
    # plt.plot(x_arr, yvals4, color="gold", label='4')
    plt.scatter(x_arr, y_arr, color='black', marker="X", label='raw data')
    plt.xlabel('packets num')
    plt.ylabel('hash')
    plt.legend(loc=4)    # 指定legend的位置右下角
    plt.title('curve_fit 1~5')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # popt, pcov = scipy.optimize.curve_fit(func_3, x, y)                # 曲线拟合，popt为函数的参数list
    # y_pred = [func_3(i, popt[0], popt[1], popt[2]) for i in x]    # 直接用函数和函数参数list来进行y值的计算

    # plot1 = plt.plot(x, y, '*', label='original values')
    # plot2 = plt.plot(x, y_pred, 'r', label='fit values')

    # plt.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))

    save_dir = 'univ1_trace/univ1_[{}-{}]_epoch{}_theta{}_x{}/[fit-num-hash]/'.format(file_begin, file_end, time_interval, theta, sample_num)
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



