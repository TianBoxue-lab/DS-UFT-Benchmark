#!/usr/bin/env python
# coding: utf-8
import os
from os.path import join as pjoin
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.mixture import GaussianMixture

sns.set(color_codes=True)


def GMM(xx, prop=2, flag='bond'):

    xx = np.array(xx)
    xx = np.expand_dims(xx, axis=1)
    gm = GaussianMixture(n_components=2, random_state=0).fit(xx)
    gmm_mean = gm.means_[0][0], gm.means_[1][0]
    gmm_std = np.sqrt(gm.covariances_[0][0][0]), np.sqrt(gm.covariances_[1][0][0])
    print(gmm_mean)
    print(gmm_std)

    if gmm_mean[0] > gmm_mean[1]:
        R_upper, R_lower = gmm_mean[0] + prop * gmm_std[0], gmm_mean[0] - prop * gmm_std[0]
        L_upper, L_lower = gmm_mean[1] + prop * gmm_std[1], gmm_mean[1] - prop * gmm_std[1]
    if gmm_mean[0] <= gmm_mean[1]:
        L_upper, L_lower = gmm_mean[0] + prop * gmm_std[0], gmm_mean[0] - prop * gmm_std[0]
        R_upper, R_lower = gmm_mean[1] + prop * gmm_std[1], gmm_mean[1] - prop * gmm_std[1]

    if flag == 'up':
        upper, lower = R_upper, R_lower
    if flag == 'low':
        upper, lower = L_upper, L_lower
    if flag == 'bond':
        upper, lower = R_upper, L_lower

    return upper, lower


def mkdirs(dirs):
    """
    Function to make directories iteratively.
    Args:
        @arg dirs: a list or a string contains the path(s) to create.
    """

    if isinstance(dirs, list):
        for dir_path in dirs:
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
    elif isinstance(dirs, str):
        if not os.path.isdir(dirs):
            os.makedirs(dirs)


def delete_wrap(file_path):
    with open(file_path, 'r') as r:
        lines = r.readlines()
    with open(file_path, 'w') as w:
        for l in lines:
            if l.strip('\n') != '':
                w.write(l)
        w.truncate()


def quick_sort(data_list):
    length = len(data_list)
    quick_sort_c(data_list, 0, length-1)


def quick_sort_c(data_list, begin, end):
    """
    Recursive function call
    """
    if begin >= end:
        return
    else:
        # Get the final index of partition data
        index = partition(data_list, begin, end)
        quick_sort_c(data_list, begin, index-1)
        quick_sort_c(data_list, index+1, end)


def partition(data_list, begin, end):
    # Select the last element as partition key
    partition_key = data_list[end]

    # index is the final position of partition key
    # Elements smaller than partition_key go to the left, larger ones go to the right
    index = begin
    for i in range(begin, end):
        if data_list[i] < partition_key:
            data_list[i], data_list[index] = data_list[index], data_list[i]
            index += 1

    data_list[index], data_list[end] = data_list[end], data_list[index]
    return index


def load_dataset_from_fasta(fasta_path: str):
    # Read sequences line by line
    # Input FASTA file, return sequences
    d_strs = []
    delete_wrap(fasta_path)

    with open(fasta_path, 'r') as infile:
        name, seq = '', ''

        while 1:
            line = infile.readline()
            line = line.strip('\n')  # Remove leading and trailing whitespace from the string
            if (line.startswith('>') or not line) and name:
                d_strs.append(seq)
            if line.startswith('>'):
                name = line[1:]
                seq = ''
            else:
                seq += line
            if not line:
                break

    return d_strs


csv_file = 'pfam_sequence_num.csv'
cv = pd.read_csv(csv_file)
cv.set_index('pfam_id', inplace=True)

# base_dir = 'D:\\Research\\8Postdoc\\TianBoxue\\pfam'  # Data path
base_dir = '/homes/Tianlab/weiming/pfam'  # Data path

# clus_coef_list = ['', '_0.5', '_0.6',  '_0.7', '_0.8', '_0.9']
pfam_aim = 'PF16492'
# g_mix = 'bond'
g_mix = None
clus_coef = '_0.9'
mkdirs(pjoin('pfam_data_process', pfam_aim))

print_log = open(pjoin('pfam_data_process', pfam_aim, pfam_aim + clus_coef + '.txt'), 'w')

file_name = pjoin(base_dir, pfam_aim, pfam_aim + clus_coef + '.fasta')
data_list = load_dataset_from_fasta(file_name)
print('before filtered, there are:\t %d seqs' % len(data_list), file=print_log)

data_no_same = list(set(data_list))  # Remove duplicates
data_no_same.sort(key=data_list.index)
length = [len(each) for each in data_no_same]
mean = np.mean(length)
std = np.std(length)

longest = max(length)
shortest = min(length)

print('longest seq length:\t\t %d' % longest, file=print_log)
print('shortest seq length:\t\t %d' % shortest, file=print_log)
print('mean seq length:\t\t %d' % mean, file=print_log)
# print(np.std(length), file=print_log)

if g_mix is None:
    upper = mean + 2 * std
    lower = mean - 2 * std
else:
    sigma3_length = [k for k in length if k <= mean + 3 * std + 1]
    upper, lower = GMM(sigma3_length, 2, g_mix)

threshold_upper = upper
threshold_lower = lower > shortest and lower or shortest
print('2 sigma upper threshold:\t %d' % threshold_upper, file=print_log)
print('2 sigma lower threshold:\t %d' % threshold_lower, file=print_log)
print('after filtered, there are:\t %d seqs' % len([k for k in length if k <= threshold_upper+1 and k >= threshold_lower-1]), file=print_log)
# print('spicial length is:\t %d seqs' % len([k for k in length if k <= 214]))

    # ax1 = sns.distplot(length,
    #                    bins=50,  # Number of bins
    #                    hist=True,  # Whether to plot histogram
#                    kde=False,
#                    color='green')
#
# ax2 = ax1.twinx()
# ax2 = sns.kdeplot(length,
#                   bw=0.25,
#                   shade=False,
#                   linestyle='-', linewidth=1.5, alpha=0.5,
#                   cut=0,
#                   color='red')
#
# pic_save_path = pjoin('pfam_data_process', pfam_aim, pfam_aim + clus_coef + '_dist_all_temp.png')
# plt.savefig(pic_save_path)
# plt.close()
# print_log.close()
#
# sigma3_length = [k for k in length if k <= mean+3*std + 1]
# # sigma3_length = [k for k in length if k <= threshold_upper]
    # ax1 = sns.distplot(sigma3_length,
    #                    bins=50,  # Number of bins
    #                    hist=True,  # Whether to plot histogram
#                    kde=False,
#                    color='green')
#
# ax2 = ax1.twinx()
# ax2 = sns.kdeplot(sigma3_length,
#                   bw=0.25,
#                   shade=False,
#                   linestyle='-', linewidth=1.5, alpha=0.5,
#                   cut=0,
#                   color='red')
#
# pic_save_path = pjoin('pfam_data_process', pfam_aim, pfam_aim + clus_coef + '_dist_3sigma_temp.png')
# plt.savefig(pic_save_path)
# plt.close()
# print_log.close()
