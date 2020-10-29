import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib
matplotlib.use('agg')

#assert(torch.__version__ <= '1.1.0')

from collections import OrderedDict
from datetime import datetime
from PIL import Image
from models_utils import * 
from data_utils import *

parser = argparse.ArgumentParser(description='BNN learning several tasks in a row, metaplasticity is controlled by the argument meta.')

parser.add_argument('--scenario', type = str, default = 'task', metavar = 'SC', help='1 mean per task or 1 mean for all task')
parser.add_argument('--net', type = str, default = 'bnn', metavar = 'NT', help='Type of net')
parser.add_argument('--in-size', type = int, default = 784, metavar = 'in', help='input size')
parser.add_argument('--hidden-layers', nargs = '+', type = int, default = [], metavar = 'HL', help='size of the hidden layers')
parser.add_argument('--out-size', type = int, default = 10, metavar = 'out', help='output size')
parser.add_argument('--task-sequence', nargs = '+', type = str, default = ['MNIST'], metavar = 'TS', help='Sequence of tasks to learn')
parser.add_argument('--lr', type = float, default = 0.005, metavar = 'LR', help='Learning rate')
parser.add_argument('--gamma', type = float, default = 1.0, metavar = 'G', help='dividing factor for lr decay')
parser.add_argument('--epochs-per-task', type = int, default = 5, metavar = 'EPT', help='Number of epochs per tasks')
parser.add_argument('--norm', type = str, default = 'bn', metavar = 'Nrm', help='Normalization procedure')
parser.add_argument('--meta', type = float, nargs = '+',  default = [0.0], metavar = 'M', help='Metaplasticity coefficients layer wise')
parser.add_argument('--rnd-consolidation', default = False, action = 'store_true', help='use shuffled Elastic Weight Consolidation')
parser.add_argument('--ewc-lambda', type = float, default = 0.0, metavar = 'Lbd', help='EWC coefficient')
parser.add_argument('--ewc', default = False, action = 'store_true', help='use Elastic Weight Consolidation')
parser.add_argument('--si-lambda', type = float, default = 0.0, metavar = 'Lbd', help='SI coefficient')
parser.add_argument('--si', default = False, action = 'store_true', help='use Synaptic Intelligence (SI)')
parser.add_argument('--bin-path', default = False, action = 'store_true', help='path integral on binary weight, else perform path integral on hidden weight')
parser.add_argument('--decay', type = float, default = 0.0, metavar = 'dc', help='Weight decay')
parser.add_argument('--init', type = str, default = 'uniform', metavar = 'INIT', help='Weight initialisation')
parser.add_argument('--init-width', type = float, default = 0.1, metavar = 'W', help='Weight initialisation width')
parser.add_argument('--save', type = bool, default = True, metavar = 'S', help='Saving the results')
parser.add_argument('--interleaved', default = False, action = 'store_true', help='saving results')
parser.add_argument('--beaker', default = False, action = 'store_true', help='use beaker')
parser.add_argument('--fb', type = float, default = 5e-3, metavar = 'fb', help='feeback coeff from last beaker to the first')
parser.add_argument('--n-bk', type = int, default = 4, metavar = 'bk', help='number of beakers')
parser.add_argument('--ratios', nargs = '+', type = float, default = [1e-2,1e-3,1e-4,1e-5], metavar = 'Ra', help='pipes specs between beakers')
parser.add_argument('--areas', nargs = '+', type = float, default = [1,2,4,8], metavar = 'Ar', help='beakers cross areas')
parser.add_argument('--device', type = int, default = 0, metavar = 'Dev', help='choice of gpu')
parser.add_argument('--seed', type = int, default = None, metavar = 'seed', help='seed for reproductibility')

args = parser.parse_args()

device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

date = datetime.now().strftime('%Y-%m-%d')
time = datetime.now().strftime('%H-%M-%S')
path = 'results/'+date+'/'+time+'_gpu'+str(args.device)
if not(os.path.exists(path)):
    os.makedirs(path)


createHyperparametersFile(path, args)

train_loader_list = []
test_loader_list = []
dset_train_list = []
task_names = []

for idx, task in enumerate(args.task_sequence):
    if task == 'MNIST':
        train_loader_list.append(mnist_train_loader)
        test_loader_list.append(mnist_test_loader)
        dset_train_list.append(mnist_dset_train)
        task_names.append(task)
    elif task == 'USPS':
        train_loader_list.append(usps_train_loader)
        test_loader_list.append(usps_test_loader)
        dset_train_list.append(usps_dset_train)
        task_names.append(task)
    elif task == 'FMNIST':
        train_loader_list.append(fashion_mnist_train_loader)
        test_loader_list.append(fashion_mnist_test_loader)
        dset_train_list.append(fmnist_dset_train)
        task_names.append(task)
    elif task == 'pMNIST':
        train_loader, test_loader, dset_train = create_permuted_loaders(task[1:])
        train_loader_list.append(train_loader)
        test_loader_list.append(test_loader)
        dset_train_list.append(dset_train)
        task_names.append(task+str(idx+1))
    elif task == 'animals':
        animals_train_loader, animals_test_loader, animals_dset_train = process_cifar10(task)
        train_loader_list.append(animals_train_loader)
        test_loader_list.append(animals_test_loader)
        dset_train_list.append(animals_dset_train)
        task_names.append('animals')
    elif task == 'vehicles':
        vehicles_train_loader, vehicles_test_loader, vehicles_dset_train = process_cifar10(task)
        train_loader_list.append(vehicles_train_loader)
        test_loader_list.append(vehicles_test_loader)
        dset_train_list.append(vehicles_dset_train)
        task_names.append('vehicles')
    elif 'cifar100' in task:
        n_subset = int(task.split('-')[1])  # task = "cifar100-20" -> n_subset = 20
        train_loader_list, test_loader_list, dset_train_list = process_cifar100(n_subset)
        task_names = ['cifar100-'+str(i+1) for i in range(n_subset)]

if args.interleaved:
    dset_train = torch.utils.data.ConcatDataset(dset_train_list)
    print(len(dset_train))
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size=100, shuffle=True, num_workers=1)
    train_loader_list = [train_loader]

    
# Hyperparameters
lr = args.lr
epochs = args.epochs_per_task
save_result = args.save
#meta = args.meta
ewc_lambda = args.ewc_lambda
si_lambda = args.si_lambda
archi = [args.in_size] + args.hidden_layers + [args.out_size]

if args.net =='bnn':
    model = BNN( archi, init = args.init, width = args.init_width, norm = args.norm).to(device)
elif args.net =='dnn':
    model = DNN( archi, init = args.init, width = args.init_width).to(device)
elif args.net=='bcnn':
    model = ConvBNN(init = args.init, width = args.init_width, norm=args.norm).to(device)

meta = {}
for n, p in model.named_parameters():
    index = int(n[9])
    p.newname = 'l'+str(index)
    if ('fc' in n) or ('cv' in n):
        meta[p.newname] = args.meta[index-1] if len(args.meta)>1 else args.meta[0]



print(model)
#plot_parameters(model, path, save=save_result)

previous_tasks_parameters = {}
previous_tasks_fisher = {}

# ewc parameters initialization
if args.ewc:
    for n, p in model.named_parameters():
        if n.find('bn') == -1: #we dont store bn parameters as we allow task dependent bn
            n = n.replace('.', '__')
            previous_tasks_fisher[n] = []
            previous_tasks_parameters[n] = [] 
elif args.si:
    W = {}
    p_prev = {}
    p_old = {}
    omega = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            W[n] = p.data.clone().zero_()
            omega[n] = p.data.clone().zero_()
            if args.net=='bnn':
                p_prev[n] = p.data.clone()  # or sign
                if args.bin_path:
                    p_old[n] = p.data.sign().clone()
                else:
                    p_old[n] = p.data.clone()
            elif args.net=='dnn':
                p_prev[n] = p.data.clone()
                p_old[n] = p.data.clone()

# Data collect initialisation
data = {}
data['net'] = args.net
data['scenario'] = args.scenario
arch = ''
if not(args.net=='bcnn'):
    for i in range(model.hidden_layers):
       arch = arch + '-' + str(model.layers_dims[i+1])

data['arch'] = arch[1:]
data['norm'] = args.norm
data['lr'], data['meta'], data['ewc'], data['SI'], data['task_order'] = [], [], [], [], []  
data['tsk'], data['epoch'], data['acc_tr'], data['loss_tr'] = [], [], [], []


for i in range(len(test_loader_list)):
    data['acc_test_tsk_'+str(i+1)], data['loss_test_tsk_'+str(i+1)] = [], []

name = '_'+data['net']+'_'+data['arch']+'_'

for t in range(len(task_names)):
    if ('cifar100' in task_names[t]) and ('cifar100' in name):
        pass
    else:
        name = name+task_names[t]+'-'

bn_states = []

lrs = [lr*(args.gamma**(-i)) for i in range(len(train_loader_list))]


if args.beaker:
    optimizer = Adam_bk(model.parameters(), lr = lr, n_bk=args.n_bk, ratios=args.ratios, areas=args.areas, feedback=args.fb, meta=meta, weight_decay=args.decay, path=path)
if args.si:
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = args.decay)

for task_idx, task in enumerate(train_loader_list):
    if not(args.beaker or args.si):
        optimizer = Adam_meta(model.parameters(), lr = lrs[task_idx], meta = meta, weight_decay = args.decay)
           
    for epoch in range(1, epochs+1):
        
        if args.ewc:
            train(model, task, task_idx, optimizer, device, args, prev_cons=previous_tasks_fisher, 
                    prev_params=previous_tasks_parameters) 
        elif args.si:
            train(model, task, task_idx, optimizer, device, args, prev_cons=omega, path_integ=W, prev_params=(p_prev, p_old) ) 
        else:
            train(model, task, task_idx, optimizer, device, args)

        data['task_order'].append(task_idx+1)
        data['tsk'].append(task_names[task_idx])
        data['epoch'].append(epoch)
        data['lr'].append(optimizer.param_groups[0]['lr'])

        train_accuracy, train_loss = test(model, task, device, verbose=True)
        
        data['acc_tr'].append(train_accuracy)
        data['loss_tr'].append(train_loss)
        data['meta'].append(meta)
        data['ewc'].append(ewc_lambda)
        data['SI'].append(si_lambda)

        current_bn_state = model.save_bn_states()
        
        for other_task_idx, other_task in enumerate(test_loader_list):

            if args.scenario == 'task':
                if other_task_idx>=task_idx:
                    model.load_bn_states(current_bn_state)
                    test_accuracy, test_loss = test(model , other_task, device, verbose=(other_task_idx==task_idx))
                else:
                    model.load_bn_states(bn_states[other_task_idx])
                    test_accuracy, test_loss = test(model , other_task, device)

            elif args.scenario =='domain':
                test_accuracy, test_loss = test(model, other_task, device, verbose=True)
            
            data['acc_test_tsk_'+str(other_task_idx+1)].append(test_accuracy)
            data['loss_test_tsk_'+str(other_task_idx+1)].append(test_loss)
        
        model.load_bn_states(current_bn_state)
    
    plot_parameters(model, path, save=save_result)
    # Uncomment for hidden weight histogram of Fig. 2g,h
    #time = datetime.now().strftime('%H-%M-%S')
    #for l in range(model.hidden_layers + 1):
    #    torch.save(model.layers['fc'+str(l+1)].weight.org.data, path+'/'+time+'_weights_fc'+str(l+1)+'.pt')
    
    bn_states.append(current_bn_state)
    if args.ewc:
        fisher = estimate_fisher(model, dset_train_list[task_idx], device, num=5000, empirical=True)
        for n, p in model.named_parameters():
            if n.find('bn') == -1: # not batchnorm
                n = n.replace('.', '__')
            
                # random consolidation
                if args.rnd_consolidation:
                    idx = torch.randperm(fisher[n].nelement())
                    previous_tasks_fisher[n].append(fisher[n].view(-1)[idx].view(fisher[n].size()))
            
                # EWC consolidation, comment when using random consolidation
                previous_tasks_fisher[n].append(fisher[n])
                previous_tasks_parameters[n].append(p.detach().clone())

    elif args.si:
        omega = update_omega(model, omega, p_prev, W)
        for n, p in model.named_parameters():
            if n.find('bn') == -1: # not batchnorm
                n = n.replace('.','__')
                if args.net=='bnn':
                    p_prev[n] = p.org.detach().clone()  # or sign
                else:
                    p_prev[n] = p.detach().clone()


time = datetime.now().strftime('%H-%M-%S')
df_data = pd.DataFrame(data)
if save_result:
    df_data.to_csv(path +'/'+time+name+'.csv', index = False)

# for Fig 5 c, d, e. Only with BNN
#total_result = switch_sign_induced_loss_increase(model, train_loader_list[0], bins=15, sample=1000, layer=1, num_run=100)    # Fig 5c
#total_result = switch_sign_induced_loss_increase(model, train_loader_list[0], bins=15, sample=2000, layer=2, num_run=100)    # Fig 5d
#total_result = switch_sign_induced_loss_increase(model, train_loader_list[0], bins=15, sample=100, layer=3, num_run=100)     # Fig 5e

# for extracting the data
#hidden_magnitude = total_result[:,:,0].mean(dim=1).view(-1)
#loss_increase = total_result[:,:,1].mean(dim=1).view(-1)
#loss_increase_std = total_result[:,:,1].std(dim=1).view(-1)

#plt.errorbar(hidden_magnitude, loss_increase, yerr=loss_increase_std, fmt='o', ms=10, mew=4)
#plt.show()

