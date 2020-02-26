import numpy as np
import pandas as pd
import math
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import argparse

assert(torch.__version__ <= '1.1.0')

from data_utils import *
from model_utils import *
from collections import OrderedDict
from datetime import datetime
from PIL import Image

os.environ['KMP_WARNINGS'] = 'off'

date = datetime.now().strftime('%Y-%m-%d')
time = datetime.now().strftime('%H-%M-%S')
path = 'results/'+date+'/'+time
if not(os.path.exists(path)):
    os.makedirs(path)

parser = argparse.ArgumentParser(description='Binary VGG with meta')

parser.add_argument('--lr', type = float, default = 0.01, metavar = 'LR', help='Learning rate')
parser.add_argument('--mbs', type = int, default = 32, metavar = 'MBS', help='MiniBatch size')
parser.add_argument('--channels', nargs = '+', type = int, default = [128,128,256,256,512,512], metavar = 'C', help='channels')
parser.add_argument('--epochs-per-task', type = int, default = 400, metavar = 'EPT', help='Number of epochs per tasks')
parser.add_argument('--meta', type = float, default = 0.0, metavar = 'M', help='Metaplasticity coefficient')
parser.add_argument('--weight-decay', type = float, default = 0.0, metavar = 'Wd', help='weight decay')
parser.add_argument('--init', type = str, default = 'gauss', metavar = 'INIT', help='Weight initialisation')
parser.add_argument('--init-width', type = float, default = 0.01, metavar = 'W', help='Weight initialisation width')
parser.add_argument('--save', type = bool, default = True, metavar = 'S', help='Saving the results')
parser.add_argument('--source', type = str, default = 'pytorch', metavar = 'Sc', help='Source of cifar10')
parser.add_argument('--nb-subset', type = int, default = 1, metavar = 'Sub', help='Number of sub datasets')
parser.add_argument('--device', type = int, default = 0, metavar = 'Dev', help='choice of gpu')

args = parser.parse_args()

device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

createHyperparametersFile(path, args)

# Hyperparameters
lr = args.lr
epochs = args.epochs_per_task
save_result = args.save
meta = args.meta
model = ConvBNN(init = args.init, width = args.init_width, channels = args.channels).to(device)
model.apply(normal_init)

print(date+'_'+time)
print(model)
plot_parameters(model, path, save=save_result)

train_loader_list, test_loader_list, full_train_loader = create_loader_list(args)

# To see that source torchvision gives always the same partition whereas source keras can be permutted
#imgs, lbls = iter(train_loader_list[0]).next()
#img = torchvision.utils.make_grid(imgs[0:args.mbs], nrow = 8)
#npimg = img.numpy()
#plt.figure(figsize=(16,9))
#plt.imshow(np.transpose(npimg, (1,2,0)))
#plt.savefig(path+'/samples_'+args.source+'.png', format = 'png')
#plt.close()


# Data collect initialisation
data = {}
data['net'] = 'BinaryCNN'
arch = 'VGG'

data['arch'] = arch
data['lr'], data['mbs'], data['meta'], data['task_order'], data['tsk'], data['epoch'] = [], [], [], [], [], []
data['acc_tr_sub'] = []
data['acc_tr'], data['acc_test'] = [], []

name = '_'+data['net']+'_'+data['arch']+'_'+str(args.nb_subset)+'CIFAR10'

optimizer = Adam_meta(model.parameters(), lr = lr, meta = meta, weight_decay = args.weight_decay)


for task_idx, task in enumerate(train_loader_list):

    for epoch in range(1, epochs+1):

        train(model, task, task_idx, optimizer, device) 
    
        data['task_order'].append(task_idx+1)
        data['tsk'].append('CIFAR10_'+str(task_idx+1)+'/'+str(args.nb_subset))
        data['epoch'].append(epoch)
        data['lr'].append(optimizer.param_groups[0]['lr'])
        data['mbs'].append(task.batch_size)
        data['meta'].append(meta)

        data['acc_tr_sub'].append(test(model, task, device, frac = args.nb_subset))
        data['acc_tr'].append(test(model, full_train_loader, device))

        test_accuracy = test(model, test_loader_list[0], device)
        data['acc_test'].append(test_accuracy)

        if (epoch%10==0) and save_result:
            time = datetime.now().strftime('%H-%M-%S')
            df_data = pd.DataFrame(data)
            df_data.to_csv(path +'/'+time+name+'_epoch_'+str(epoch)+'.csv', index = False)
            plot_parameters(model, path, save=save_result)


time = datetime.now().strftime('%H-%M-%S')
df_data = pd.DataFrame(data)
if save_result:
    df_data.to_csv(path +'/'+time+name+'.csv', index = False)

torch.save({'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict() },  path + '/checkpoint.tar')
