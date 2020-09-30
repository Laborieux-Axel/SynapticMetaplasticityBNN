import numpy as np
import math
import pandas as pd
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('Agg')

from collections import OrderedDict
from datetime import datetime

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

                decay = torch.max(0.3*torch.abs(p.data), torch.ones_like(p.data))
                decayed_exp_avg = torch.mul(torch.ones_like(p.data)-torch.pow(torch.tanh(group['meta']*torch.abs(p.data)),2) ,exp_avg)
                exp_avg_2 = torch.div(exp_avg, decay)
  
                if len(p.size())==1: # True if p is bias, false if p is weight
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    #p.data.addcdiv_(-step_size, exp_avg , denom)  #normal update
                    p.data.addcdiv_(-step_size, torch.where(condition_consolidation, decayed_exp_avg, exp_avg) , denom)  #assymetric lr for metaplasticity
                    
        return loss


class Adam_bk(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), n_bk=1, ratios=[0], meta = 0.0, feedback=0.0, eps=1e-8,
                 weight_decay=0, amsgrad=False, path='.'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, n_bk=n_bk, ratios=ratios, meta=meta, feedback=feedback, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, path=path)
        super(Adam_bk, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_bk, self).__setstate__(state)
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
            n_bk = group['n_bk']
            ratios = group['ratios']
            meta = group['meta']
            feedback = group['feedback']
            path = group['path']

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
                    # Initializing beakers
                    for bk_idx in range(n_bk+1):
                        if bk_idx==n_bk:  # create an additional beaker clamped at 0
                            state['bk'+str(bk_idx)+'_t-1'] = torch.zeros_like(p)
                            state['bk'+str(bk_idx)+'_t']   = torch.zeros_like(p)
                        else:             # create other beakers at equilibrium
                            state['bk'+str(bk_idx)+'_t-1'] = torch.empty_like(p).copy_(p)
                            state['bk'+str(bk_idx)+'_t']   = torch.empty_like(p).copy_(p)

                        state['bk'+str(bk_idx)+'_lvl'] = []

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)


                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)  #p.data

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

                if p.dim()==1: # True if p is bias, false if p is weight
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    # weight update
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                    p.data.add_(ratios[0]*(state['bk1_t-1']-state['bk0_t-1']))
                    p.data.add_(torch.where( (state['bk'+str(n_bk-1)+'_t-1'] - state['bk0_t-1']) * state['bk'+str(n_bk-1)+'_t-1'].sign() > 0 , feedback*(state['bk'+str(n_bk-1)+'_t-1'] - state['bk0_t-1']),
                                                                                                                      torch.zeros_like(p.data)))
                    # Update of the beaker levels
                    with torch.no_grad():
                        for bk_idx in range(1, n_bk):
                        # diffusion entre les bk dans les deux sens + metaplasticité sur le dernier                                
                            if bk_idx==(n_bk-1):
                                condition = (state['bk'+str(bk_idx-1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1'])*state['bk'+str(bk_idx)+'_t-1'] < 0
                                decayed_m = 1 - torch.tanh(meta*state['bk'+str(bk_idx)+'_t-1'])**2
                                state['bk'+str(bk_idx)+'_t'] = torch.where(condition, state['bk'+str(bk_idx)+'_t-1'] + ratios[bk_idx-1]*decayed_m*(state['bk'+str(bk_idx-1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1']) + ratios[bk_idx]*(state['bk'+str(bk_idx+1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1']),
                                                                                      state['bk'+str(bk_idx)+'_t-1'] + ratios[bk_idx-1]*(state['bk'+str(bk_idx-1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1']) + ratios[bk_idx]*(state['bk'+str(bk_idx+1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1']))
                            else:
                                state['bk'+str(bk_idx)+'_t'] = state['bk'+str(bk_idx)+'_t-1'] + ratios[bk_idx-1]*(state['bk'+str(bk_idx-1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1']) + ratios[bk_idx]*(state['bk'+str(bk_idx+1)+'_t-1'] - state['bk'+str(bk_idx)+'_t-1'])


                # Plotting beaker levels and distributions
                fig = plt.figure(figsize=(12,9))
                for bk_idx in range(n_bk):
                    if bk_idx==0:
                        state['bk'+str(bk_idx)+'_t-1'] = p.data
                    else:
                        state['bk'+str(bk_idx)+'_t-1'] = state['bk'+str(bk_idx)+'_t']

                    if p.size() == torch.empty(1024,1024).size():
                        state['bk'+str(bk_idx)+'_lvl'].append(state['bk'+str(bk_idx)+'_t-1'][11, 100].detach().item())
                        if state['step']%600==0:
                            plt.plot(state['bk'+str(bk_idx)+'_lvl'])
                            fig.savefig(path + '/trajectory.png', fmt='png', dpi=300)
                plt.close()

                if p.dim()!=1 and state['step']%600==0:
                    fig2 = plt.figure(figsize=(12,9))
                    for bk_idx in range(n_bk):
                        plt.hist(state['bk'+str(bk_idx)+'_t-1'].detach().cpu().numpy().flatten(), 100, label='bk'+str(bk_idx), alpha=0.5)
                    plt.legend()
                    fig2.savefig(path+'/bk_'+str(bk_idx)+'_'+str(p.size(0))+'-'+str(p.size(1))+'.png', fmt='png')
                    plt.close()


        return loss





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

class BNN(torch.nn.Module):
    """
    MyNet can consist either of fc layers followed by batchnorm, fc weights being either float kind="classical_bn" 
    or binarized kind="binary", or fc layers with biases kind="classical_bias". When BatchNorm is used the adtication function is 
    the sign function and when biases are used the activation function is Tanh
    weights can be initialized to gaussian with init="gauss" or uniform distribution with init="uniform"
    The width of the distribution is tuned with width
    the only non specified argument is the list of neurons [input, hidden ... , output]
    """
    def __init__(self, layers_dims, init = "gauss", width = 0.01):
        super(BNN, self).__init__()
        
        self.hidden_layers = len(layers_dims)-2
        self.layers_dims = layers_dims 
        
        layer_list = []

        for layer in range(self.hidden_layers+1): 
            layer_list = layer_list + [(  ('fc'+str(layer+1) ) , BinarizeLinear(layers_dims[layer], layers_dims[layer+1], bias = False)) ]
            layer_list = layer_list + [(  ('bn'+str(layer+1) ) , torch.nn.BatchNorm1d(layers_dims[layer+1], affine = False, track_running_stats = True)) ]
                       
        self.layers = torch.nn.ModuleDict(OrderedDict( layer_list ))
        
        #weight init
        for layer in range(self.hidden_layers+1): 
            if init == "gauss":
                torch.nn.init.normal_(self.layers['fc'+str(layer+1)].weight, mean=0, std=width)
            if init == "uniform":
                torch.nn.init.uniform_(self.layers['fc'+str(layer+1)].weight, a= -width/2, b=width/2)
            
    def forward(self, x):

        size = self.layers_dims[0]
        x = x.view(-1, size)
        
        for layer in range(self.hidden_layers+1):
            x = self.layers['fc'+str(layer+1)](x)
            x = self.layers['bn'+str(layer+1)](x)
            if layer != self.hidden_layers:
                x = SignActivation.apply(x)
        return x
                   
def plot_parameters(model, path, save=True):
    
    hid_lay = model.hidden_layers
    num_hist = hid_lay + 1
    fig = plt.figure(figsize=(15, num_hist*5))

    for hist_idx in range(num_hist):
        fig.add_subplot(num_hist,2,2*hist_idx+1)
        if not hasattr(model.layers['fc'+str(hist_idx+1)].weight, 'org'):
            weights = model.layers['fc'+str(hist_idx+1)].weight.data.cpu().numpy()
        else:
            weights = np.copy(model.layers['fc'+str(hist_idx+1)].weight.org.data.cpu().numpy())
        plt.title('Poids de la couche '+str(hist_idx+1))
        plt.hist( weights.flatten(), 100, density = True)
        fig.add_subplot(num_hist, 2, 2*hist_idx+2)
    
    #plt.show()
    if save:
        time = datetime.now().strftime('%H-%M-%S')
        fig.savefig(path+'/'+time+'_weight_distribution.png')
    plt.close()
    
def train(model, train_loader, current_task_index, optimizer, device, criterion = torch.nn.CrossEntropyLoss(), clamp = 10, verbose = False):
    
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

    

def test(model, data_loader, device, frac = 1, criterion = torch.nn.CrossEntropyLoss(), verbose = False):
    
    model.eval()
    test_loss = 0
    correct = 0
    
    for data, target in data_loader:
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(data_loader.dataset)
    
    test_acc = round( 100. * float(correct) * frac / len(data_loader.dataset)  , 2)
    
    if verbose :
        print('Test accuracy: {}/{} ({:.2f}%)'.format(
            correct, len(data_loader.dataset),
            test_acc))
    
    return test_acc
