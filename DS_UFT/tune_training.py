#!/usr/bin/env python
# coding: utf-8
import torch
import esm
from torch import nn
import d2l.torch
import math
import os
from os.path import join as pjoin
from prepare_data import load_data_human, mkdirs, load_data_affinity
from data_parallel import BalancedDataParallel

# torch.set_printoptions(threshold=np.inf)

class EsmForMaskedLM(nn.Module):
    def __init__(self, model):
        super(EsmForMaskedLM, self).__init__()
        self.esm2model = model  # Load ESM model, ESM has built-in MLM MLP function

    def forward(self, tokens_id, mlm_positions):
        logits = self.esm2model(tokens_id, return_contacts=False)["logits"]  # (batch_size, seq_len, vocab_size)

        num_pred_positions = mlm_positions.shape[1]
        positions_serial = mlm_positions.reshape(-1)
        batch_size = tokens_id.shape[0]
        batch_id = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_id, num_pred_positions)
        # Assume batch_size=2, num_pred_positions=3
        # Then batch_idx is np.array([0,0,0,1,1,1])

        return logits[batch_idx, positions_serial]  # (batch_size*num_pred_positions, vocab_size)


# Calculate the forward propagation loss for a batch
def _get_batch_loss_mlm(net, loss, tokens_id_X, mlm_positions_X, mlm_weights_X, mlm_positions_Y):
    # Forward propagation
    mlm_positions_Y_hat = net(tokens_id_X, mlm_positions_X)

    # Calculate masked language model loss
    mlm_loss = loss(mlm_positions_Y_hat, mlm_positions_Y.reshape(-1)) * mlm_weights_X.reshape(-1)  # Both are 1D vectors, element-wise multiplication
    mean_mlm_loss = mlm_loss.sum() / (mlm_weights_X.sum() + 1e-8)  # Calculate mean MLM loss for a batch: represents average loss per predicted token

    return mean_mlm_loss


def train_esm2(train_iter, net, loss, devices, num_steps, out_model):
    """for restart model"""
    if RESTART:
        net.esm2model.load_state_dict(torch.load(out_model))

    net = nn.DataParallel(module=net, device_ids=devices).to(devices[0])
    # net.to(devices)
    # net = BalancedDataParallel(2, module=net, device_ids=devices).to(devices[0])
    optim = torch.optim.Adam(params=net.parameters(), lr=4e-5)  # 1e-4 1e-5
    step = 0
    # Masked language model loss
    accumulator = d2l.torch.Accumulator(3)  # For accumulating sums over `n` variables.
    # animator = d2l.torch.Animator(xlabel='step', ylabel='loss', xlim=[1, num_steps], legend=['mlm_loss'])  # For plotting data in animation.
    timer = d2l.torch.Timer()
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for (tokens_id_X, mlm_positions_X, mlm_weights_X, mlm_positions_Y) in train_iter:
            tokens_id_X = tokens_id_X.to(devices[0])
            mlm_positions_X = mlm_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_positions_Y = mlm_positions_Y.to(devices[0])
            optim.zero_grad()
            timer.start()
            mlm_loss = _get_batch_loss_mlm(net, loss, tokens_id_X, mlm_positions_X, mlm_weights_X, mlm_positions_Y)
            mlm_loss.backward()
            optim.step()
            accumulator.add(mlm_loss, tokens_id_X.shape[0], 1)
            timer.stop()
            # animator.add(step + 1, accumulator[0]/accumulator[2])

            step += 1
            if step and step % 10000 == 0:
                torch.save(net.module.esm2model.state_dict(), out_model)

            if step == num_steps:
                num_steps_reached = True
                break

            if step and step % 1000 == 0:
                print('mlm_loss:', accumulator[0] / accumulator[2], accumulator[1] / timer.sum(), ' data/s on', devices)

    print('mlm_loss:', accumulator[0] / accumulator[2], accumulator[1] / timer.sum(), 'data/s on', devices)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    RESTART = False

    # file_base_dir = '/homes/Tianlab/weiming/pfam'
    # model_base_dir = '/homes/Tianlab/weiming/tune_model_0.9'  # tune_model_0.9 tune_model_0.5

    # file_base_dir = '/homes/Tianlab/weiming/clans/2'
    # model_base_dir = '/homes/Tianlab/weiming/tune_model_0.9_clans'  # tune_model_0.9 tune_model_0.5

    file_base_dir = '/homes/Tianlab/weiming/combine/'
    model_base_dir = '/homes/Tianlab/weiming/tune_model_combine'  # tune_model_0.9 tune_model_0.5

    pfam_aim = 'combine'
    clus_coef = '_0.5'
    g_mix = None
    # g_mix = 'bond'
    # g_mix = 'low'

    # file_name = pjoin(file_base_dir, pfam_aim, pfam_aim + clus_coef + '.fasta')
    #
    # mkdirs(pjoin(model_base_dir, pfam_aim))
    # out_model = pjoin(model_base_dir, pfam_aim, pfam_aim + clus_coef + '.pth')

    file_name = pjoin(file_base_dir, pfam_aim + clus_coef, pfam_aim + clus_coef + '.fasta')

    mkdirs(pjoin(model_base_dir, pfam_aim + clus_coef))
    out_model = pjoin(model_base_dir, pfam_aim + clus_coef, pfam_aim + clus_coef + '.pth')

    batch_size, num_workers, truncation_seq_length = 16, 16, 1024

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    loss = nn.CrossEntropyLoss(reduction='none')
    net = EsmForMaskedLM(model)
    devices = d2l.torch.try_all_gpus()

    train_iter, data_size = load_data_human(file_name, batch_size, num_workers, truncation_seq_length, g_mix)
    num_steps = int(math.ceil(data_size * 20 / batch_size / 10000) * 10000)
    print('data path:\t %s' % (pfam_aim + clus_coef + '.fasta'))
    print('batch size:\t %d' % batch_size)
    print('step num:\t %d' % num_steps)
    print('fine tune all layers')
    train_esm2(train_iter, net, loss, devices, num_steps=num_steps, out_model=out_model)