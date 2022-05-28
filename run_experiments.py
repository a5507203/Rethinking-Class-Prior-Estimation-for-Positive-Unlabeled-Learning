import os
import time
import importlib
import json
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random
import shutil
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
import torch.nn.functional as F
import matlab.engine

from model.classification_models import simple_FCNet

from util.averagemeter import *
from util.data_loader import get_loader
from util.torch_dataset import DatasetArray
from util.dataset_util import *

from baselines_interface.kmpms import run_kmpms
from baselines_interface.en import run_en
from baselines_interface.alphamax import run_alphamax
from baselines_interface.roc import run_roc
from baselines_interface.dedpul import run_dedpul
from baselines_interface.rpg import run_rankpruning
def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def parse_args():
    parser = argparse.ArgumentParser()

    # run config
    parser.add_argument('--dataset', type=str, default='') 
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--pretrained', type=str2bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--sample_size', type=int, default=400)
    parser.add_argument('--relabel_frac', type=float, default=0.01)  
    parser.add_argument('--val_set_frac', type=float, default=0.2)  
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)

    # optim config
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--repeat', type=int, default=10)

    args = parser.parse_args()

    optim_config = OrderedDict([
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
        ('momentum', args.momentum)
    ])

    data_config = OrderedDict([
        ('dataset', args.dataset)
    ])
    # some run_configs are not initialized here, their value will be assigned automically later, e.g.. positive_frac, positive_label, seed
    run_config = OrderedDict([
        ('outdir', args.outdir),
        ('num_workers', args.num_workers),
        ('pretrained',args.pretrained),
        ('sample_size',args.sample_size),
        ('relabel_frac',args.relabel_frac),
        ('val_set_frac',args.val_set_frac),
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        # AS MENTIONED IN OUR PAPER: we select $\{0.25, 0.5, 0.75\}$ fraction of either positive (or negative) examples
        # to be the sample of the component distribution $H$. We let the rest of the examples to be the 
        # sample of the mixture distribution $F$
        ('positive_fracs',[0.25, 0.5, 0.75]),
        ('positive_labels',[0,1]),
        # for each sample size, $10$ repeated experiments are carried out with random sampling
        ("num_random_experiments",args.repeat),
    ])

    config = OrderedDict([
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config)
    ])

    return config


def load_model(input_dim):

    Network = simple_FCNet(input_dim=input_dim)

    return Network



def train(epoch, model, optimizer, criterion, train_loader):
    global global_step
   
    top1_acc_meter = AverageMeter()
    top2_acc_meter = AverageMeter()
    ce_loss_meter = AverageMeter()

    model.train()
    start = time.time()

    for step, (data, targets) in enumerate(train_loader):
        global_step += 1

        data = data.to(device)
        targets = targets.to(device)
        outputs = model(data)
        #forward
        loss = criterion(outputs, targets)
        batch_size = data.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        top1_acc, top2_acc = accuracy(outputs.data, targets.data, topk=(1,2))
        top1_acc_meter.update(top1_acc.item(), batch_size)
        top2_acc_meter.update(top2_acc.item(), batch_size)
        ce_loss_meter.update(loss.item(), batch_size)



def validate_and_test(epoch, model, criterion, val_loader, optim_config, is_test = False):


    top1_acc_meter = AverageMeter()
    top2_acc_meter = AverageMeter()
    #cross entropy loss meter and regularization loss meter
    ce_loss_meter = AverageMeter()

    #swith model to to eval mode
    model.eval()
    start = time.time()
    for step, (data, targets) in enumerate(val_loader):

        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(data)

        ce_loss = criterion(outputs, targets)
        num = data.size(0)        
        
        top1_acc, top2_acc = accuracy(outputs.data, targets.data, topk=(1, 2))
        top1_acc_meter.update(top1_acc.item(), num)
        top2_acc_meter.update(top2_acc.item(), num)
        ce_loss_meter.update(ce_loss.item(), num)


    top1_acc_avg = top1_acc_meter.avg
    top2_acc_avg = top2_acc_meter.avg


    return top1_acc_avg, top2_acc_avg



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 



def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    


def adjust_learning_rate(optimizer, epoch, baselr, model_type=1):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch > 30:
        lr = baselr * 0.01
    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



def generate_new_inputs(model, tr_unlabeled_sample, relabel_frac):

    dataset = DatasetArray(data = tr_unlabeled_sample)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
        num_workers=1,
    )

    # swith model to to eval mode
    model.eval()
    start = time.time()
   
    p = []
    for step, (data, targets) in enumerate(loader):

        data = data.to(device)
        labels = targets.numpy()
        # get the posterior probabilites
        with torch.no_grad():
            outputs = model(data)

        oh_labels = one_hot_embedding(targets,2).numpy()
        probs = F.softmax(outputs, dim=1).cpu().data.numpy()
        data = data.cpu().numpy()
        p += probs[:,0].tolist()

    s = cus_sort(p)
 
    return get_anchor_index(s,relabel_frac)

def cus_sort(l):
    d = {i:l[i] for i in range(len(l))}
    s = [(k,d[k]) for k in sorted(d, key=d.get, reverse=False)]
    return s

def get_anchor_index(index_p_list,relabel_frac):

    n = len(index_p_list)
    num_anchors = int(n*relabel_frac)
    min_p = index_p_list[0][1]
    max_p = index_p_list[-1][1]
    min_f_list = []
    max_f_list = []
    for (idx, p) in index_p_list:
        if(len(min_f_list)<num_anchors):
            min_f_list.append(idx)
        else:
            break
    for (idx, p) in reversed(index_p_list):
        if(len(max_f_list)<num_anchors):
            max_f_list.append(idx)
        else:
            break
    return min_f_list, max_f_list



# The sample size is fixed as input arguement, thus 3*10*2 exmperiments are conducted for a dataset with the same sample size.
# Totally, as mentioned in the paper, for a dataset, 3*10*2*3 exmperiments are conducted by setting the sample size as [800,1600,3200].
def main():
    
    global device
    global best_top1_acc
    global global_step

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #configs 
    config = parse_args()
    run_config = config['run_config']
    optim_config = config['optim_config']
    total_sample_length =  run_config['sample_size']
    error = {"AM":0. , "RAM":0. , "ROC":0. , "RROC":0. , "EN":0. , "REN":0. , "KM1":0. , "RKM1":0. , "KM2":0. , "RKM2":0. , "DPL":0. , "RDPL":0. , "RPG":0. , "RRPG":0. }

    # AS MENTIONED IN OUR PAPER: 
    # we select $\{0.25, 0.5, 0.75\}$ fraction of 
    # either positive (or negative) examples to be the sample of the positive distribution $P_p$. 
    # We let the rest of the examples to be the sample of the unlabeled distribution $P_x$.
    for positive_frac in run_config['positive_fracs']:
        # for each sample size, $10$ repeated experiments are carried out with random sampling.
        for repeat_times in range(run_config['num_random_experiments']):
            randm_seed = random.randint(1, 10000)
            for positive_label in run_config['positive_labels']:
            
                global_step = 0
                best_top1_acc = 0
                best_top2_acc = 0

                #config
                dataset = config["data_config"]["dataset"]
                run_config['positive_frac'] = positive_frac
                run_config['positive_label'] = positive_label
                run_config['seed'] = randm_seed
                run_config['outdir'] = './trained_model/'+dataset+'/'+'_ss'+str(total_sample_length)+'_cf'+str(positive_frac)+'rt'+str(randm_seed)+'_cl'+str(positive_label)
                outdir = run_config['outdir']
                print("")
                print("___________________________________\n")
                # create output directory
                create_dir(run_config['outdir'])

                # set random seed
                seed_torch(run_config['seed'])
                # data loaders
        
                train_loader, val_loader, tr_unlabeled_sample, tr_positive_sample, val_unlabeled_sample, val_positive_sample, class_prior, _ = get_loader(
                    positive_label = run_config['positive_label'],
                    batch_size = run_config['batch_size'], 
                    num_workers = run_config['num_workers'],
                    dataset_path="./datasets/"+dataset+".data", 
                    total_sample_length=run_config['sample_size'],
                    positive_frac=run_config['positive_frac'],
                    val_set_frac = run_config['val_set_frac']
                )
                # model
                model = load_model(input_dim = len(tr_positive_sample[0])-1)
                model.to(device)
                #criterion and optimizer
                criterion = nn.CrossEntropyLoss(size_average=True)

                optimizer = torch.optim.SGD(
                    params = model.parameters(),
                    lr=optim_config['base_lr'], 
                    momentum=optim_config['momentum'],
                    weight_decay=optim_config['weight_decay']
                )

                # double check if the trained model exists
                if (not run_config["pretrained"]) or (not is_check_point(outdir)):
    
                    for epoch in range(1, run_config['epochs'] + 1):

                        train(epoch, model, optimizer, criterion, train_loader)
                        top1_acc_avg, top2_acc_avg = validate_and_test(epoch, model, criterion, val_loader, optim_config)
                        
                        #save model
                        state = OrderedDict([
                            ('config', config),
                            ('state_dict', model.state_dict()),
                            ('optimizer', optimizer.state_dict()),
                            ('epoch', epoch),
                            ('top1-accuracy', top1_acc_avg),
                            ('top2-accuracy', top2_acc_avg),
                        ])
                    
                        model_path = os.path.join(outdir, 'model_state.pth')
                        torch.save(state, model_path)
                        if (top1_acc_avg > best_top1_acc):
                            best_model_path = os.path.join(outdir, 'model_best.pth')
                            shutil.copyfile(model_path, best_model_path)
                            best_top1_acc = top1_acc_avg
                            best_top2_acc = top2_acc_avg


                error = runALGS(tr_unlabeled_sample, tr_positive_sample, outdir, class_prior, run_config["relabel_frac"], error,model)
         

    print("")    
    print("++++summary+++"+str(total_sample_length)+"+") 
    for alg, err in error.items():
        print(str(alg) +", " +str(err/60))
    print("++++++++")
    print("")


def is_check_point(outdir):
    return os.path.exists(outdir+'/model_best.pth')
    

def create_dir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)


#run each baseline and its regrouping version
def runALGS(tr_unlabeled_sample, tr_positive_sample, outdir, class_prior, relabel_frac, error, model):

    print("kappa star= "+str(class_prior))

    AM = 1
    RAM = 1

    ROC = 1
    RROC = 1

    EN = 1
    REN = 1

    KM1 = 1
    RKM1 = 1

    KM2 = 1
    RKM2 = 1

    eng.clearM(nargout=0)
    saved_dict = torch.load(outdir+'/model_best.pth')
    print("top1-accuracy:", saved_dict["top1-accuracy"])
    model.load_state_dict(saved_dict["state_dict"])

    min_f_idxs, max_f_idxs = generate_new_inputs(model, tr_unlabeled_sample, relabel_frac)
   
    tr_unlabeled_sample=np.array(tr_unlabeled_sample)[:,:-1].tolist()
    tr_positive_sample = np.array(tr_positive_sample)[:,:-1].tolist()
    estimated_set_A = [ tr_unlabeled_sample[idx] for idx in min_f_idxs ]


    #run alphamax (AM) and regrouping-alphamax (RAM)
    AM = run_alphamax(eng,x=tr_unlabeled_sample,x1=tr_positive_sample)
    RAM = run_alphamax(eng,x=tr_unlabeled_sample,x1=tr_positive_sample+estimated_set_A)
    error["AM"] += abs(class_prior-AM)
    print("AM_estimate= "+str(AM))
    error["RAM"] += abs(class_prior-RAM)
    print("RAM_estimate= "+str(RAM))
    eng.clearM(nargout=0)

    #run ROC  and regrouping-ROC (RROC)
    ROC = run_roc(eng,x=tr_unlabeled_sample,x1=tr_positive_sample)
    RROC = run_roc(eng,x=tr_unlabeled_sample,x1=tr_positive_sample+estimated_set_A)
    error["ROC"] += abs(class_prior-ROC)
    print("ROC_estimate= "+str(ROC))
    error["RROC"] += abs(class_prior-RROC)
    print("RROC_estimate= "+str(RROC))
    eng.clearM(nargout=0)

    #run EN and regrouping-EN (REN)
    EN = run_en(eng,x=tr_unlabeled_sample,x1=tr_positive_sample)
    REN = run_en(eng,x=tr_unlabeled_sample,x1=tr_positive_sample+estimated_set_A)
    error["EN"] += abs(class_prior-EN)
    print("EN_estimate= "+str(EN))
    error["REN"] += abs(class_prior-REN)
    print("REN_estimate= "+str(REN))
    eng.clearM(nargout=0)

    #run KM1, KM2 and regrouping-KM1 (RKM1), regrouping-KM2 (RKM2)
    KM1, KM2 = run_kmpms(tr_unlabeled_sample,tr_positive_sample)
    RKM1, RKM2 = run_kmpms(tr_unlabeled_sample,tr_positive_sample+estimated_set_A)
    
    error["KM1"] += abs(class_prior-KM1)
    print ("KM1_estimate= "+str(KM1))
    error["RKM1"] += abs(class_prior-RKM1)
    print ("RKM1_estimate= "+str(RKM1))

    error["KM2"] += abs(class_prior-KM2)
    print ("KM2_estimate= "+str(KM2))
    error["RKM2"] += abs(class_prior-RKM2)
    print ("RKM2_estimate= "+str(RKM2))


    #run DEDPUL (DPL) and regrouping-DEDPUL (RDPL)
    DPL = run_dedpul(x=tr_unlabeled_sample,x1=tr_positive_sample)
    RDPL = run_dedpul(x=tr_unlabeled_sample,x1=tr_positive_sample+estimated_set_A)
    error["DPL"] += abs(class_prior-DPL)
    print("DPL_estimate= "+str(DPL))
    error["RDPL"] += abs(class_prior-RDPL)
    print("RDPL_estimate= "+str(RDPL))

    #run Rankpruning (RPG) and regrouping-Rankpruning (RRPG)
    RPG = run_rankpruning(x=tr_unlabeled_sample,x1=tr_positive_sample)
    RRPG = run_rankpruning(x=tr_unlabeled_sample,x1=tr_positive_sample+estimated_set_A)
    error["RPG"] += abs(class_prior-RPG)
    print("RPG_estimate= "+str(RPG))
    error["RRPG"] += abs(class_prior-RRPG)
    print("RRPG_estimate= "+str(RRPG))

    
    return error



device = None
global_step = 0
best_top1_acc = 0
best_top2_acc = 0

eng = matlab.engine.start_matlab(background=False)
eng.cd("./baselines_matlab")

if __name__ == '__main__':
    main()
