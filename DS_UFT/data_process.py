import numpy as np
from sklearn.mixture import GaussianMixture


def GMM(xx, prop=2, flag='bond'):
    """
    Use Gaussian Mixture Model to determine upper and lower bounds in length distribution
    """

    xx = np.array(xx)
    xx = np.expand_dims(xx, axis=1)
    gm = GaussianMixture(n_components=2, random_state=0).fit(xx)
    gmm_mean = gm.means_[0][0], gm.means_[1][0]
    gmm_std = np.sqrt(gm.covariances_[0][0][0]), np.sqrt(gm.covariances_[1][0][0])

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


def delete_data_out(data_list, prop, g_mix):
    data_no_same = list(set(data_list))  # Remove duplicates
    data_no_same.sort(key=data_list.index)

    length = [len(each) for each in data_no_same]
    mean = np.mean(length)
    std = np.std(length)
    if g_mix is None:
        upper = mean + prop * std
        lower = mean - prop * std
    else:
        sigma3_length = [k for k in length if k <= mean + 3 * std + 1]
        upper, lower = GMM(sigma3_length, prop, g_mix)

    for i in range(len(data_no_same) - 1, -1, -1):
        if len(data_no_same[i]) >= upper + 1:
            del data_no_same[i]
        elif len(data_no_same[i]) <= lower - 1:
            del data_no_same[i]

    return data_no_same
