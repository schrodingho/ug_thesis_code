# train for ug

import torch
import argparse
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
# from torch.utils.tensorboard import SummaryWriter
from models import MedRec
from encoders import Encoder
from aggregators import MeanAggregator

from util import llprint, multi_label_metric, get_n_params

torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'MedRec'
resume_name = ''
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')

args = parser.parse_args()
model_name = args.model_name
resume_name = args.resume_path

def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    for step, input in enumerate(data_eval):#验证集输入
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm_idx, adm in enumerate(input):
            # old:
            # target_output1 = model(input[:adm_idx+1])#输入模型，得到output
            # new
            target_output1= model(input=input[:adm_idx + 1], patient_step=step)
            y_gt_tmp = np.zeros(voc_size[2])#y
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))


    llprint('\tJaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    # dill.dump(obj=smm_record, file=open('../data/records.pkl', 'wb'))
    # dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb')
    # print('avg med', med_cnt / visit_cnt)

    return np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'

    device = torch.device('cuda:0')

    data = dill.load(open(data_path, 'rb'))

    voc = dill.load(open(voc_path, 'rb'))
    train_adj=np.load('../data/ps_adj.npy')
    adj_dict = defaultdict(set)
    for i, row in enumerate(train_adj):
        for j, col in enumerate(row):
            if col == 1:
                adj_dict[i].add(j)
    train_adj=adj_dict

    feat_med=np.load('../data/feat_med.npy')

    f_med = nn.Embedding(feat_med.shape[0],feat_med.shape[1])
    f_med.weight = nn.Parameter(torch.FloatTensor(feat_med), requires_grad=False)

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    agg_med= MeanAggregator(f_med, cuda=True)
    enc_med= Encoder(f_med, feat_med.shape[1], voc_size[2], train_adj, agg_med, gcn=True, cuda=True)
    enc_med.num_samples = 5

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    EPOCH = 40
    LR = 0.001
    TEST = args.eval


    model = MedRec(voc_size, enc_med , emb_dim=32, device=device)
    if TEST:
        model.load_state_dict(torch.load(open(resume_name, 'rb')))
    model.to(device=device)

    print('parameters', get_n_params(model))
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

    if TEST:
        eval(model, data_test, voc_size, 0)
    else:
        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            prediction_loss_cnt = 0#pre
            neg_loss_cnt = 0
            for step, input in enumerate(data_train):#第一层：每个病人
                for idx, adm in enumerate(input):#第二层：一个病人的时序
                    seq_input = input[:idx+1] #

                    # writer.add_graph(model, verbose=False)

                    loss1_target = np.zeros((1, voc_size[2]))#(1,153)
                    loss1_target[:, adm[2]] = 1
                    target_output1=model(input=seq_input, patient_step=step)
                    # only Loss1
                    loss = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))

                    # writer.add_scalar('Train_Loss', loss, epoch)
                    # writer.add_scalar('gateh', model.parameters()['gate_trans'], epoch)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))

            ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)

            history['ja'].append(ja)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)
            history['loss'].append(loss_record1)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            # torch.save(model.state_dict(), open(os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f.model' % (epoch, ja)), 'wb'))
            print('')
            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja

        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))
        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved', model_name, 'train_ug.model'), 'wb'))
        print('best_epoch:', best_epoch)

if __name__ == '__main__':
    # writer = SummaryWriter('./saved/visualization/log/')
    main()
