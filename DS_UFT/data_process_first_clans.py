#!/usr/bin/env python
# coding: utf-8
import os
from os.path import join as pjoin
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

sns.set(color_codes=True)


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
    """
    Remove extra '\n' from file
    """
    with open(file_path, 'r') as r:
        lines = r.readlines()
    with open(file_path, 'w') as w:
        for l in lines:
            if l.strip('\n') != '':
                w.write(l)
        w.truncate()


def quick_sort(data_list):
    """
    Quick sort
    """
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

base_dir = '/homes/Tianlab/weiming/clans/1'  # Data path
# clus_coef_list = ['', '_0.5', '_0.6',  '_0.7', '_0.8', '_0.9']
real_names = [dI for dI in os.listdir(base_dir) if os.path.isdir(pjoin(base_dir, dI))]
for pfam_aim in tqdm(real_names):

    # if cv.loc[pfam_aim, 'pfam_cluster_0.5_num'] <= 5000:  # Only output pfam_cluster_0.5_num > 5000
    #     continue

    clus_coef = '_0.9'

    mkdirs(pjoin('clans_data_process', pfam_aim))
    print_log = open(pjoin('clans_data_process', pfam_aim, pfam_aim + clus_coef + '.txt'), 'w')


    file_name = pjoin(base_dir, pfam_aim, pfam_aim + clus_coef + '.fasta')
    print(file_name)
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

    threshold_upper = mean+2*std
    threshold_lower = mean-2*std > shortest and mean-2*std or shortest
    print('2 sigma upper threshold:\t %d' % threshold_upper, file=print_log)
    print('2 sigma lower threshold:\t %d' % threshold_lower, file=print_log)
    print('after filtered, there are:\t %d seqs' % len([k for k in length if
                                                        threshold_upper + 1 >= k >= threshold_lower - 1]), file=print_log)

    sns.histplot(length,
                 bins=50,  # Number of bins
                 # hist=True,  # Whether to plot histogram
                 kde=True,
                 kde_kws = dict(bw_adjust=2,
                                cut=0,),
                 line_kws =dict(ls='-', lw=1.5,),
                 color='green',
                 alpha=0.3,)
    pic_save_path = pjoin('clans_data_process', pfam_aim, pfam_aim + clus_coef + '_dist_all.png')
    plt.savefig(pic_save_path)
    plt.close()

    # Considering that some data may be too long, making it hard to see the distribution in plots,
    # use 3sigma on the right side of length distribution as threshold for plotting
    sigma3_length = [k for k in length if k <= mean+3*std + 1]
    sns.histplot(sigma3_length,
                 bins=50,  # Number of bins
                 # hist=True,  # Whether to plot histogram
                 kde=True,
                 kde_kws = dict(bw_adjust=2,
                                cut=0,),
                 line_kws =dict(ls='-', lw=1.5,),
                 color='green',
                 alpha=0.3,)
    pic_save_path = pjoin('clans_data_process', pfam_aim, pfam_aim + clus_coef + '_dist_3sigma.png')
    plt.savefig(pic_save_path)
    plt.close()
    print_log.close()
