import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from torch.utils.data import random_split
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.activation_based import functional, surrogate, neuron
import torchbnn as bnn
from copy import deepcopy
from spikingjelly.activation_based import layer
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import pickle


def main():
    parser = argparse.ArgumentParser(description='Train SNNs using Pytorch')
    parser.add_argument('-data-dir', type=str, default=r' ',
                        help='root dir of DVS Gesture dataset')
    parser.add_argument('-device', default=' ', help='device')
    parser.add_argument('--home', default=r" ")
    parser.add_argument('--dataset', default=r" ")
    args = parser.parse_args()
    print(args)

    def softmax(x):
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
        softmax_probs = exp_x / sum_exp_x
        return softmax_probs

    dtype = torch.float
    device = torch.device("cpu")
    batch_size = 128
    dataset_path = args.home + args.dataset
    digits = [i for i in range(10)]
    input_size = [2, 26, 26]
    num_inputs = 2 * 26 * 26
    num_hidden = 1000
    num_outputs = 11
    num_steps = 80
    beta = 0.95
    n_val = 7000
    num_ter = 4
    n_validation = 50
    n_test = 288 - n_validation
    batch_size_validation = n_validation
    batch_size_test = n_test
    target_size = 3
    num_ensemble_train = 3
    r = 45
    mappings = ((torch.arange(num_ter) + 1) * (num_steps / num_ter)).int()

    class DVSGestureNet(torch.nn.Module):
        def __init__(self, channels=128):  # 4-100-3
            super(DVSGestureNet, self).__init__()
            self.fc1 = bnn.BayesConv2d(prior_mu=0, prior_sigma=0.03, in_channels=2, out_channels=channels,
                                       kernel_size=3, padding=1, bias=False)
            self.nor1 = bnn.BayesBatchNorm2d(prior_mu=0, prior_sigma=0.03, num_features=channels)
            self.lif1 = neuron.LIFNode()
            self.max1 = layer.MaxPool2d(2, 2)

            self.fc2 = bnn.BayesConv2d(prior_mu=0, prior_sigma=0.03, in_channels=channels, out_channels=channels,
                                       kernel_size=3, padding=1, bias=False)
            self.nor2 = bnn.BayesBatchNorm2d(prior_mu=0, prior_sigma=0.03, num_features=channels)
            self.lif2 = neuron.LIFNode()
            self.max2 = layer.MaxPool2d(2, 2)

            self.fc3 = bnn.BayesConv2d(prior_mu=0, prior_sigma=0.03, in_channels=channels, out_channels=channels,
                                       kernel_size=3, padding=1, bias=False)
            self.nor3 = bnn.BayesBatchNorm2d(prior_mu=0, prior_sigma=0.03, num_features=channels)
            self.lif3 = neuron.LIFNode()
            self.max3 = layer.MaxPool2d(2, 2)

            self.fc4 = bnn.BayesConv2d(prior_mu=0, prior_sigma=0.03, in_channels=channels, out_channels=channels,
                                       kernel_size=3, padding=1, bias=False)
            self.nor4 = bnn.BayesBatchNorm2d(prior_mu=0, prior_sigma=0.03, num_features=channels)
            self.lif4 = neuron.LIFNode()
            self.max4 = layer.MaxPool2d(2, 2)

            self.fc5 = bnn.BayesConv2d(prior_mu=0, prior_sigma=0.03, in_channels=channels, out_channels=channels,
                                       kernel_size=3, padding=1, bias=False)
            self.nor5 = bnn.BayesBatchNorm2d(prior_mu=0, prior_sigma=0.03, num_features=channels)
            self.lif5 = neuron.LIFNode()
            self.max5 = layer.MaxPool2d(2, 2)

            self.flat = layer.Flatten()
            self.drop1 = layer.Dropout(0.5)
            self.fc6 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.03, in_features=channels * 4 * 4, out_features=512)
            self.lif6 = neuron.LIFNode()
            self.drop2 = layer.Dropout(0.5)
            self.fc7 = layer.Linear(512, 110)
            self.lif7 = neuron.LIFNode()
            self.vote = layer.VotingLayer(10)

        def forward(self, xt):
            result = []
            for t in range(num_steps):
                x = self.max1(self.lif1(self.nor1(self.fc1(xt[t]))))
                x = self.max2(self.lif2(self.nor2(self.fc2(x))))
                x = self.max1(self.lif3(self.nor3(self.fc3(x))))
                x = self.max4(self.lif4(self.nor4(self.fc4(x))))
                x = self.max5(self.lif5(self.nor5(self.fc5(x))))
                y = self.vote(self.lif7(self.fc7(self.drop2(self.lif6(self.fc6(self.drop1(self.flat(x))))))))
                result.append(y.unsqueeze(0))

            return torch.cat(result, 0)

    net = DVSGestureNet()

    loss = nn.CrossEntropyLoss(reduction='none')
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.001

    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=80, split_by='number')
    val_test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=80,
                                 split_by='number')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
    num_epochs = 1

    for epoch in range(num_epochs):
        train_batch = iter(train_loader)

        for data, targets in train_batch:
            data = data.transpose(0, 1).to(device) 
            targets = targets.to(device)  

            net.train()

            mem_rec_av = torch.zeros(num_steps, targets.size(0), num_outputs)
            for i in range(num_ensemble_train):
                spk_rec, mem_rec, _ = net(data)
                mem_rec_av = mem_rec_av + mem_rec
                functional.reset_net(net)

            mem_rec_av = mem_rec_av / num_ensemble_train

            loss_tr = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_tr += loss(mem_rec_av[step], targets).sum() / targets.size(0)

            kl = kl_loss(net)
            cost = loss_tr + kl_weight * kl

            optimizer.zero_grad()
            loss_tr.backward()
            optimizer.step()

    aver_length = 10
    inner_length = 20
    datasets = {}
    for i in range(aver_length):
        val_dataset, test_dataset = random_split(val_test_set, [n_val, n_test])
        datasets[i] = {'val': val_dataset, 'test': test_dataset}

    n_para = 4
    
    av_gcov_pmg = torch.zeros(n_para, aver_length)
    av_gdl_pmg = torch.zeros(n_para, aver_length)
    
    av_gcov_ncav = torch.zeros(n_para, aver_length)
    av_gdl_ncav = torch.zeros(n_para, aver_length)

    count = 0
    alpha = 0.1
    for num_ensemble in [1, 3, 5, 10]:

        for outav in range(aver_length):
            
            inner_gcov_pmg = 0
            inner_gdl_pmg = 0
            
            inner_gcov_ncav = 0
            inner_gdl_ncav = 0
            for av in range(inner_length):
                
                val_dataset = datasets[av + outav * inner_length]['val']
                subset = Subset(val_dataset, range(n_validation))
                test_dataset = datasets[av + outav * inner_length]['test']
                test_loader = torch.utils.data.DataLoader(
                    dataset=test_dataset,
                    batch_size=batch_size_test,
                    shuffle=False,
                    drop_last=False,
                    num_workers=2,
                    pin_memory=True
                )

                val_loader = torch.utils.data.DataLoader(
                    dataset=subset,
                    batch_size=batch_size_validation,
                    shuffle=True,
                    drop_last=False,
                    num_workers=2,
                    pin_memory=True
                )

                
                gcov_pmg = 0
                gdelay_pmg = 0
                
                gcov_ncav = 0
                gdelay_ncav = 0

                with torch.no_grad():
                    net.eval()

                    pvalue_fg = torch.zeros(num_ter, n_test, num_outputs)
                    spk_val_av = torch.zeros(num_steps, n_validation, num_outputs)
                    spk_test_av = torch.zeros(num_steps, n_test, num_outputs)
                    for i in range(num_ensemble):
                        val_data = iter(val_loader)
                        test_data = iter(test_loader)
                        
                        NCg_pmg = torch.zeros(num_ter, n_validation + 1)
                        NCg_pmg[:, 0] = 99999

                        for data_val, targets_val in val_data:
                            data_val = data_val.transpose(0, 1).to(device)  
                            targets_val = targets_val.to(device) 

                            spk_val, _, _ = net(data_val) 

                            spk_val_av = spk_val_av + spk_val

                            
                            one_hot_matrix = torch.zeros(n_validation, num_outputs)
                            one_hot_matrix.scatter_(1, targets_val.long().unsqueeze(1), 1)
                            NCg_batch = torch.zeros(num_ter, n_validation)
                            for t in range(num_ter):
                                NCg_batch[t] = torch.diag(torch.mm(
                                    -torch.log(softmax(spk_val[:int((num_steps / num_ter) * (t + 1))].sum(dim=0))),
                                    one_hot_matrix.transpose(0, 1)))
                                NCg_pmg[t, 1:] = NCg_batch[t]

                            functional.reset_net(net)

                        for data, targets in test_data:
                            data = data.transpose(0, 1).to(device) 
                            targets = targets.to(device) 

                            spk_test, _, spk_hidden = net(data)  
                            spk_test_av = spk_test_av + spk_test

                            spk_ter = torch.zeros(num_ter, n_test, num_outputs)
                            for t in range(num_ter):
                                spk_ter[t] = spk_test[:int((num_steps / num_ter) * (t + 1)), :, :].sum(dim=0)

                            
                            NCg_test_pmg = torch.zeros(num_ter, n_test, num_outputs)
                            for i in range(num_outputs):
                                for t in range(num_ter):
                                    possible = i * torch.ones([n_test]).to(torch.uint8)
                                    one_hot_matrix_test = torch.zeros(possible.size(0), num_outputs)  
                                    one_hot_matrix_test.scatter_(1, possible.long().unsqueeze(1), 1)
                                    NCg_test_pmg[t][:, i] = torch.diag(torch.mm(-torch.log(softmax(spk_ter[t])),
                                                                                one_hot_matrix_test.transpose(0,
                                                                                                              1)))  

                            
                            pvalue_g = torch.zeros(num_ter, n_test, num_outputs)
                            for t in range(num_ter):
                                comparison_g = NCg_test_pmg[t].unsqueeze(2) <= NCg_pmg[t]
                                pvalue_g[t] = comparison_g.sum(dim=2) / (n_validation + 1)

                            functional.reset_net(net)


                        pvalue_fg = pvalue_fg + torch.pow(pvalue_g, r)

                    
                    spk_val_av = spk_val_av / num_ensemble

                    
                    NCg_ncav = torch.zeros(num_ter, n_validation + 1)
                    NCg_ncav[:, 0] = 99999
                    one_hot_matrix = torch.zeros(n_validation, num_outputs)
                    one_hot_matrix.scatter_(1, targets_val.long().unsqueeze(1), 1)  
                    NCg_batch = torch.zeros(num_ter, n_validation)
                    for t in range(num_ter):
                        NCg_batch[t] = torch.diag(
                            torch.mm(-torch.log(softmax(spk_val_av[:int((num_steps / num_ter) * (t + 1))].sum(dim=0))),
                                     one_hot_matrix.transpose(0, 1)))
                        NCg_ncav[t, 1:] = NCg_batch[t]

                    spk_test_av = spk_test_av / num_ensemble
                    spk_ter = torch.zeros(num_ter, n_test, num_outputs)
                    for t in range(num_ter):
                        spk_ter[t] = spk_test_av[:int((num_steps / num_ter) * (t + 1)), :, :].sum(dim=0)

                    NCg_test_ncav = torch.zeros(num_ter, n_test, num_outputs)
                    for i in range(num_outputs):
                        for t in range(num_ter):
                            possible = i * torch.ones([n_test]).to(torch.uint8)
                            one_hot_matrix_test = torch.zeros(possible.size(0), num_outputs) 
                            one_hot_matrix_test.scatter_(1, possible.long().unsqueeze(1), 1)
                            NCg_test_ncav[t][:, i] = torch.diag(torch.mm(-torch.log(softmax(spk_ter[t])),
                                                                         one_hot_matrix_test.transpose(0,
                                                                                                       1)))  

                    pvalue_ncav = torch.zeros(num_ter, n_test, num_outputs)
                    for t in range(num_ter):
                        comparison_g = NCg_test_ncav[t].unsqueeze(2) <= NCg_ncav[t]
                        pvalue_ncav[t] = comparison_g.sum(dim=2) / (n_validation + 1)

                    carg_matrix_ncav = ((num_ter * pvalue_ncav > alpha) + 0).sum(dim=2).transpose(0,
                                                                                                  1)  

                   
                    mask = (carg_matrix_ncav <= target_size) + 0
                    zeros = mask.sum(dim=1)
                    
                    idx = torch.where(zeros == 0)[0]
                    mask[idx, num_ter - 1] = 1
                    stop_time_ncav = torch.argmax((mask == 1) + 0, dim=1) 

                    
                    delayg_batch = mappings[stop_time_ncav] 
                    gdelay_ncav += delayg_batch.sum()

                   
                    pvalue_gt = pvalue_ncav.transpose(0, 1)
                    mask_sm = torch.zeros_like(pvalue_gt, dtype=torch.bool)
                    for i in range(stop_time_ncav.shape[0]):
                        mask_sm[i, stop_time_ncav[i], :] = True
                    selected_pvalue = pvalue_gt[mask_sm].reshape(stop_time_ncav.shape[0], -1)  
                    for i in range(num_outputs):
                        if i == 0:
                            
                            pre_sm = ((num_ter * selected_pvalue > alpha) + 0)[:, i:i + 1] * (-1) + 1
                            pree_sm = torch.where(pre_sm == 1, torch.tensor(10), pre_sm)
                        else:
                            
                            pre_sm = ((num_ter * selected_pvalue > alpha) + 0)[:, i:i + 1] * i
                            pree_sm = torch.where(pre_sm == 0, torch.tensor(10), pre_sm)
                        gcov_ncav += (pree_sm == torch.unsqueeze(targets, 1)).sum().item()

                   
                    pvalue_fg = torch.pow(pvalue_fg / num_ensemble, 1 / r) * torch.pow(torch.tensor(num_ensemble),
                                                                                       1 / r)

                    carg_matrix = ((num_ter * pvalue_fg > alpha) + 0).sum(dim=2).transpose(0, 1)  

                    
                    mask = (carg_matrix <= target_size) + 0
                    zeros = mask.sum(dim=1)
                   
                    idx = torch.where(zeros == 0)[0]
                    mask[idx, num_ter - 1] = 1
                    stop_timeg = torch.argmax((mask == 1) + 0, dim=1)  

                    
                    delayg_batch = mappings[stop_timeg]  
                    gdelay_pmg += delayg_batch.sum()

                    pvalue_gt = pvalue_fg.transpose(0, 1)  
                    mask_sm = torch.zeros_like(pvalue_gt, dtype=torch.bool)
                    for i in range(stop_timeg.shape[0]):
                        mask_sm[i, stop_timeg[i], :] = True
                    selected_pvalue = pvalue_gt[mask_sm].reshape(stop_timeg.shape[0], -1) 
                    for i in range(num_outputs):
                        if i == 0:
                            
                            pre_sm = ((num_ter * selected_pvalue > alpha) + 0)[:, i:i + 1] * (-1) + 1
                            pree_sm = torch.where(pre_sm == 1, torch.tensor(10), pre_sm)
                        else:
                            
                            pre_sm = ((num_ter * selected_pvalue > alpha) + 0)[:, i:i + 1] * i
                            pree_sm = torch.where(pre_sm == 0, torch.tensor(10), pre_sm)
                        gcov_pmg += (pree_sm == torch.unsqueeze(targets, 1)).sum().item()

                
                inner_gcov_pmg = inner_gcov_pmg + gcov_pmg / n_test
                inner_gdl_pmg = inner_gdl_pmg + gdelay_pmg / n_test

                inner_gcov_ncav = inner_gcov_ncav + gcov_ncav / n_test
                inner_gdl_ncav = inner_gdl_ncav + gdelay_ncav / n_test

            av_gcov_pmg[count, outav] = inner_gcov_pmg / inner_length 
            av_gdl_pmg[count, outav] = inner_gdl_pmg / inner_length

            av_gcov_ncav[count, outav] = inner_gcov_ncav / inner_length  
            av_gdl_ncav[count, outav] = inner_gdl_ncav / inner_length

        count = count + 1

    
    gcov_final_pmg = av_gcov_pmg.sum(dim=1) / aver_length
    gdl_final_pmg = av_gdl_pmg.sum(dim=1) / aver_length

    
    std_cov_pmg = np.zeros(n_para)
    std_dl_pmg = np.zeros(n_para)
    for i in range(n_para):
        std_cov_pmg[i] = torch.std(av_gcov_pmg[i]).numpy()
        std_dl_pmg[i] = torch.std(av_gdl_pmg[i]).numpy()

   
    gcov_pmg_np = gcov_final_pmg.numpy()
    gdl_pmg_np = gdl_final_pmg.numpy()

    
    gcov_final_ncav = av_gcov_ncav.sum(dim=1) / aver_length
    gdl_final_ncav = av_gdl_ncav.sum(dim=1) / aver_length

    
    std_cov_ncav = np.zeros(n_para)
    std_dl_ncav = np.zeros(n_para)
    for i in range(n_para):
        std_cov_ncav[i] = torch.std(av_gcov_ncav[i]).numpy()
        std_dl_ncav[i] = torch.std(av_gdl_ncav[i]).numpy()

    
    gcov_ncav_np = gcov_final_ncav.numpy()
    gdl_ncav_np = gdl_final_ncav.numpy()

    text_dict = {'gcov_pmg_np': gcov_pmg_np, 'gdl_pmg_np': gdl_pmg_np, 'std_cov_pmg': std_cov_pmg,
                 'std_dl_pmg': std_dl_pmg, 'std_cov_ncav': std_cov_ncav, 'std_dl_ncav': std_dl_ncav}

    with open('BSpikeCP_K.pickle', 'wb') as f:
        pickle.dump(text_dict, f)


if __name__ == '__main__':
    main()