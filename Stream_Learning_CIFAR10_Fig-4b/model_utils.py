import numpy as np
import pandas as pd
import math
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from collections import OrderedDict
from datetime import datetime
from PIL import Image


class SignActivation(torch.autograd.Function): # We define a sign activation with derivative equal to clip

    @staticmethod
    def forward(ctx, i):
        result = i.sign()
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        grad_i = grad_output.clone()
        grad_i[i.abs() > 1.0] = 0
        return grad_i

def Binarize(tensor):
    return tensor.sign()

class BinarizeLinear(torch.nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        
    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = torch.nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(torch.nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = torch.nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
   
def normal_init(m):
    if m.__class__.__name__.find('Binarize') !=-1:
        torch.nn.init.xavier_normal_(m.weight)
    #elif m.__class__.__name__.find('BatchNorm') !=-1:
    #    torch.nn.init.ones_(m.weight)


class ConvBNN(torch.nn.Module):

    def __init__(self, init = "gauss", width = 0.01, channels = [128,128,256,256,512,512]):
        super(ConvBNN, self).__init__()
        
        # input: (mb x 3 x 32 x 32)
        self.features = torch.nn.Sequential(BinarizeConv2d(3, channels[0], kernel_size=3, padding=1, bias=False), #out: (mb x channels[0] x 32 x 32)
                                            torch.nn.BatchNorm2d(channels[0], affine=True, track_running_stats=True),
                                            torch.nn.Hardtanh(inplace=True),
                                            
                                            BinarizeConv2d(channels[0],channels[1], kernel_size=3, padding=1, bias=False), #out: (mb x channels[1] x 32 x 32)
                                            torch.nn.MaxPool2d(kernel_size=2), #out: (mb x channels[1] x 16 x 16)
                                            torch.nn.BatchNorm2d(channels[1], affine=True, track_running_stats=True),
                                            torch.nn.Hardtanh(inplace=True),

                                            BinarizeConv2d(channels[1],channels[2],kernel_size=3, padding=1, bias=False), #out: (mb x channels[2] x 16 x 16)
                                            torch.nn.BatchNorm2d(channels[2], affine=True, track_running_stats=True),
                                            torch.nn.Hardtanh(inplace=True),

                                            BinarizeConv2d(channels[2],channels[3],kernel_size=3, padding=1, bias=False), #out: (mb x channels[3] x 16 x 16)
                                            torch.nn.MaxPool2d(kernel_size=2), #out: (mb x channels[3] x 8 x 8)
                                            torch.nn.BatchNorm2d(channels[3], affine=True, track_running_stats=True),
                                            torch.nn.Hardtanh(inplace=True),
        
                                            BinarizeConv2d(channels[3],channels[4],kernel_size=3, padding=1, bias=False), #out: (mb x channels[4] x 8 x 8)
                                            torch.nn.BatchNorm2d(channels[4], affine=True, track_running_stats=True),
                                            torch.nn.Hardtanh(inplace=True),

                                            BinarizeConv2d(channels[4],channels[5],kernel_size=3, bias=False), #out: (mb x channels[5] x 6 x 6)
                                            torch.nn.MaxPool2d(kernel_size=2), #out: (mb x channels[5] x 3 x 3)
                                            torch.nn.BatchNorm2d(channels[5], affine=True, track_running_stats=True),
                                            torch.nn.Hardtanh(inplace=True),
                                            )   
        
        self.classifier = torch.nn.Sequential(BinarizeLinear(channels[5]*9,2048, bias=False),
                                              torch.nn.BatchNorm1d(2048, affine=True, track_running_stats=True),
                                              torch.nn.Hardtanh(inplace=True),
                                              torch.nn.Dropout(0.5),

                                              BinarizeLinear(2048,2048, bias=False),
                                              torch.nn.BatchNorm1d(2048, affine=True, track_running_stats=True),
                                              torch.nn.Hardtanh(inplace=True),
                                              torch.nn.Dropout(0.5),

                                              BinarizeLinear(2048,10, bias=False),
                                              torch.nn.BatchNorm1d(10, affine=True, track_running_stats=True),
                                             )        
        

    def forward(self, x):
        
        x = self.features(x)       
        x = x.view(x.size(0), -1)
        x = self.classifier(x)      

        return x

class Adam_meta(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), meta = 0.75, eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, meta=meta, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_meta, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_meta, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
            
                    if len(p.size())!=1:
                        state['followed_weight'] = np.random.randint(p.size(0)),np.random.randint(p.size(1))
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])


                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                

                binary_weight_before_update = torch.sign(p.data)
                condition_consolidation = (torch.mul(binary_weight_before_update,exp_avg) > 0.0 )

                decayed_exp_avg = torch.mul(torch.ones_like(p.data)-torch.pow(torch.tanh(group['meta']*torch.abs(p.data)),2) ,exp_avg)

  
                if len(p.size())==1: # True if p is bias, false if p is weight
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    #p.data.addcdiv_(-step_size, exp_avg , denom)  #normal update
                    p.data.addcdiv_(-step_size, torch.where(condition_consolidation, decayed_exp_avg, exp_avg) , denom)  #assymetric lr for metaplasticity
                    
        return loss


def train(model, train_loader, current_task_index, optimizer, device, criterion = torch.nn.CrossEntropyLoss(), verbose = False):
    
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # This loop is for BNN parameters having 'org' attribute
        for p in list(model.parameters()): # blocking weights with org value greater than a threshold by setting grad to 0 
            if hasattr(p,'org'):
                p.data.copy_(p.org)
                
        optimizer.step()
        
        # This loop is only for BNN parameters as they have 'org' attribute
        for p in list(model.parameters()):  # updating the org attribute
            if hasattr(p,'org'):
                p.org.copy_(p.data)
    

def test(model, test_loader, device, frac = 1, criterion = torch.nn.CrossEntropyLoss(), verbose = False):
    
    model.eval()
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    
    test_acc = round( 100. * float(correct) * frac / len(test_loader.dataset)  , 2)
    
    if verbose :
        print('Test accuracy: {}/{} ({:.2f}%)'.format(
            correct, len(test_loader.dataset),
            test_acc))
    
    return test_acc

def plot_parameters(model, path, save=True):
    
    fig = plt.figure(figsize=(15, 30))
    i = 1

    for (n, p) in model.named_parameters():
        
        if (n.find('bias') == -1) and (len(p.size()) != 1):  #bias or batchnorm weight -> no plot
            fig.add_subplot(8,2,i)
            if model.__class__.__name__.find('B') != -1:  #BVGG -> plot p.org
                if hasattr(p,'org'):
                    weights = p.org.data.cpu().numpy()
                else:
                    weights = p.data.cpu().numpy()
                binet = 100
            else:
                weights = p.data.cpu().numpy()            #TVGG or FVGG plot p
                binet = 50
            i+=1
            plt.title( n.replace('.','_') )
            plt.hist( weights.flatten(), binet)

    if save:
        time = datetime.now().strftime('%H-%M-%S')
        fig.savefig(path+'/'+time+'_weight_distribution.png')
    plt.close()




