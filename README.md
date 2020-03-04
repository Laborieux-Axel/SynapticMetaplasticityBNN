# Synaptic Metaplasticity in Binarized Neural Networks

This repository contains the code producing the figures of the paper Synaptic Metaplasticity in Binarized Neural Networks (BNN). To set the environment run in your conda main environment:  
> conda config --add channels conda-forge  
> conda create --name environment_name --file requirements.txt  
> conda activate environment_name  
> conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch

In each folder except Quadratic Binary Task, model_utils.py contains all classes and functions relevant to the model architectures and the train/test phase.
data_utils.py contains functions relevant to data management. All the simulations produce csv file with accuracies and losses tracked at every epoch.  
The code for BNN modules was adapted from https://github.com/itayhubara/BinaryNet.pytorch  
The code for EWC was adapted from https://github.com/GMvandeVen/continual-learning  

## Continual Learning

#### BNN with increasing metaplasticity  

These runs produce the data of Fig. 2 (a), (b), (c), (d), (e).  To obtain the data of Fig. 2(g), (h), lines 170-172 in main.py need to be uncommented.  

> python main.py --net bnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 0.0 --epochs-per-task 40 --task-sequence pMNIST pMNIST pMNIST pMNIST pMNIST pMNIST  
> python main.py --net bnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 0.5 --epochs-per-task 40 --task-sequence pMNIST pMNIST pMNIST pMNIST pMNIST pMNIST  
> python main.py --net bnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 1.0 --epochs-per-task 40 --task-sequence pMNIST pMNIST pMNIST pMNIST pMNIST pMNIST  
> python main.py --net bnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 1.35 --epochs-per-task 40 --task-sequence pMNIST pMNIST pMNIST pMNIST pMNIST pMNIST  

#### BNN with elastic weight consolidation  

For random consolidation of Table 1, some lines need to be commented in main.py  
> python main.py --net bnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --ewc-lambda 5000.0 --epochs-per-task 40 --task-sequence pMNIST pMNIST pMNIST pMNIST pMNIST pMNIST  

#### Full Precision Neural Network with increasing metaplasticity

These runs produce the data of Fig. 2(f).  

> python main.py --net dnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 0.0 --epochs-per-task 40 --task-sequence pMNIST pMNIST pMNIST pMNIST pMNIST pMNIST  
> python main.py --net dnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 0.5 --epochs-per-task 40 --task-sequence pMNIST pMNIST pMNIST pMNIST pMNIST pMNIST  
> python main.py --net dnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 1.0 --epochs-per-task 40 --task-sequence pMNIST pMNIST pMNIST pMNIST pMNIST pMNIST  
> python main.py --net dnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 1.35 --epochs-per-task 40 --task-sequence pMNIST pMNIST pMNIST pMNIST pMNIST pMNIST  

#### BNN learning MNIST-FMNIST  

These runs produce the data of Fig. 3(a), (b), (c), (d).

> python main.py --net bnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 0.0 --epochs-per-task 50 --task-sequence MNIST FMNIST  
> python main.py --net bnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 1.5 --epochs-per-task 50 --task-sequence MNIST FMNIST  
> python main.py --net bnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 0.0 --epochs-per-task 50 --task-sequence FMNIST MNIST  
> python main.py --net bnn --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 1.5 --epochs-per-task 50 --task-sequence FMNIST MNIST  


#### Switching the sign of binary weights in a BNN  

Lines at the end of main.py need to be uncommented to produce the data of Fig. 5(c), (d), (e).  

> python main.py --net bnn --hidden-layers 1024 1024 --decay 1e-7 --meta 1.35 --lr 0.005 --epochs-per-task 40 --task-sequence MNIST

## Stream Learning (Fashion MNIST)  

The data of Fig. 4(a) is produced by the following runs:  
> python main.py --task FMNIST --hidden-layers 1024 1024 --lr 0.005 --decay 1e-7 --meta 2.5 --epochs-per-task 20 --nb-subset 1  
> python main.py --task FMNIST --hidden-layers 1024 1024 --lr 0.005 --decay 1e-7 --meta 0.0 --epochs-per-task 20 --nb-subset 1  
> python main.py --task FMNIST --hidden-layers 1024 1024 --lr 0.005 --decay 1e-7 --meta 2.5 --epochs-per-task 20 --nb-subset 60  
> python main.py --task FMNIST --hidden-layers 1024 1024 --lr 0.005 --decay 1e-7 --meta 0.0 --epochs-per-task 20 --nb-subset 60  

## Stream Learning (CIFAR-10)  

The data of Fig. 4(b) is produced by the following runs:  
> python main.py --lr 0.0001 --mbs 64 --source keras --nb-subset 1 --epochs-per-task 200 --meta 0.0 --init gauss --init-width 0.007  
> python main.py --lr 0.0001 --mbs 64 --source keras --nb-subset 20 --epochs-per-task 200 --meta 0.0 --init gauss --init-width 0.007  
> python main.py --lr 0.0001 --mbs 64 --source keras --nb-subset 1 --epochs-per-task 200 --meta 13.0 --init gauss --init-width 0.007  
> python main.py --lr 0.0001 --mbs 64 --source keras --nb-subset 20 --epochs-per-task 200 --meta 13.0 --init gauss --init-width 0.007  

## Quadratic Binary Task  

The Quadratic Binary Problem is a jupyter notebook with Fig. 5(a), (b) already produced inside.

