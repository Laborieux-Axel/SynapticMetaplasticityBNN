# Synaptic Metaplasticity in Binarized Neural Networks

This repository contains the code producing the figures of the __[paper](https://arxiv.org/abs/2003.03533)__ "Synaptic Metaplasticity in Binarized Neural Networks" (BNNs). A GPU is needed to run the simulations in a reasonable time. To set the environment run in your conda main environment (5 minutes needed):  
```
conda config --add channels conda-forge  
conda create --name environment_name --file requirements.txt  
conda activate environment_name  
conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
```

In each folder except Quadratic Binary Task, model_utils.py contains all classes and functions relevant to the model architectures and the train/test phase.
data_utils.py contains functions relevant to data management. All the simulations produce csv file with accuracies and losses tracked at every epoch.  
The code for BNN modules was adapted from [this repo](https://github.com/itayhubara/BinaryNet.pytorch).  
The code for EWC was adapted from [this repo](https://github.com/GMvandeVen/continual-learning).  

## Continual Learning

#### BNN with increasing metaplasticity  

These runs (~3 hours) produce the data of Fig. 2 (a), (b), (c), (d), (e).  To obtain the data of Fig. 2(g), (h), lines 170-172 in main.py need to be uncommented.  
```
#Fig. 2(a)
python main.py --net 'bnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 0.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'  
```
```
#Fig. 2(b)
python main.py --net 'bnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 0.5 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'  
```
```
#Fig. 2(c)
python main.py --net 'bnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 1.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'  
```
```
#Fig. 2(d)
python main.py --net 'bnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 1.35 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'  
```

#### BNN with elastic weight consolidation  

For random consolidation (~3 hours) of Table 1, some lines need to be commented in main.py  
```
python main.py --net 'bnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --ewc-lambda 5000.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'  
```

#### Full Precision Neural Network with increasing metaplasticity

These runs (~3 hours) produce the data of Fig. 2(f).  

```
python main.py --net 'dnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 0.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'  
python main.py --net 'dnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 0.5 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'  
python main.py --net 'dnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 1.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'  
python main.py --net 'dnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 1.35 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'  
```


#### Interleaved training

```
#Fig 3(a-b)
python main.py --net 'bnn' --hidden-layers 512 512 --lr 0.005 --decay 1e-8 --meta 1.35 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
python main.py --net 'bnn' --hidden-layers 1024 1024 --lr 0.005 --decay 1e-8 --meta 1.35 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
python main.py --net 'bnn' --hidden-layers 2048 2048 --lr 0.005 --decay 1e-8 --meta 1.35 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
python main.py --net 'bnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-8 --meta 1.35 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'

python main.py --net 'bnn' --hidden-layers 512 512 --lr 0.005 --decay 1e-8 --ewc --ewc-lambda 5e3 --meta 0.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
python main.py --net 'bnn' --hidden-layers 1024 1024 --lr 0.005 --decay 1e-8 --ewc --ewc-lambda 5e3 --meta 0.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
python main.py --net 'bnn' --hidden-layers 2048 2048 --lr 0.005 --decay 1e-8 --ewc --ewc-lambda 5e3 --meta 0.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
python main.py --net 'bnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-8 --ewc --ewc-lambda 5e3 --meta 0.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
```

```
#Fig 3(c)
python main.py --net 'bnn' --interleaved --hidden-layers 512 512 --lr 0.005 --decay 1e-8 --meta 0.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
python main.py --net 'bnn' --interleaved --hidden-layers 1024 1024 --lr 0.005 --decay 1e-8 --meta 0.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
python main.py --net 'bnn' --interleaved --hidden-layers 2048 2048 --lr 0.005 --decay 1e-8 --meta 0.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
python main.py --net 'bnn' --interleaved --hidden-layers 4096 4096 --lr 0.005 --decay 1e-8 --meta 0.0 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
```

```
#Fig 3(d)
python main.py --net 'bnn' --interleaved --hidden-layers 512 512 --lr 0.005 --decay 1e-8 --meta 1.35 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
python main.py --net 'bnn' --interleaved --hidden-layers 1024 1024 --lr 0.005 --decay 1e-8 --meta 1.35 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
python main.py --net 'bnn' --interleaved --hidden-layers 2048 2048 --lr 0.005 --decay 1e-8 --meta 1.35 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
python main.py --net 'bnn' --interleaved --hidden-layers 4096 4096 --lr 0.005 --decay 1e-8 --meta 1.35 --epochs-per-task 40 --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
```

#### BNN learning MNIST-FMNIST  

These runs (~1 hour) produce the data of Fig. 3(a), (b), (c), (d).

```
#Fig. 4(a)
python main.py --net 'bnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 0.0 --epochs-per-task 50 --task-sequence MNIST FMNIST
```
```
#Fig. 4(b)
python main.py --net 'bnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 1.5 --epochs-per-task 50 --task-sequence MNIST FMNIST
```
```
#Supp. Fig. 6(c)
python main.py --net 'bnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 0.0 --epochs-per-task 50 --task-sequence FMNIST MNIST
```
```
#Supp. Fig. 6(d)
python main.py --net 'bnn' --hidden-layers 4096 4096 --lr 0.005 --decay 1e-7 --meta 1.5 --epochs-per-task 50 --task-sequence FMNIST MNIST  
```

#### BNN learning MNIST-USPS

```
#Fig. 4(c)
python MNIST-USPS/main.py --net 'bcnn' --ch 1 12 20 30 --kr 4 4 4 --decay 1e-8 --norm 'bn' --lr 0.005 --meta 1.2 --epochs-per-task 240 --task-sequence 'MNIST' 'USPS'
python MNIST-USPS/main.py --net 'bcnn' --ch 1 6 10 15 --kr 4 4 4 --norm 'bn' --lr 0.005 --meta 0.0 --epochs-per-task 240 --task-sequence 'USPS'
python MNIST-USPS/main.py --net 'bcnn' --ch 1 6 10 15 --kr 4 4 4 --norm 'bn' --lr 0.005 --meta 0.0 --epochs-per-task 240 --task-sequence 'MNIST'
```

```
#Fig. 4(d)
python MNIST-USPS/main.py --net 'bcnn' --ch 1 10 15 20 --kr 4 4 4 --decay 1e-8 --norm 'bn' --lr 0.005 --meta 1.2 --epochs-per-task 240 --task-sequence 'MNIST' 'USPS'
python MNIST-USPS/main.py --net 'bcnn' --ch 1 6 10 15 --kr 4 4 4 --norm 'bn' --lr 0.005 --meta 0.0 --epochs-per-task 240 --task-sequence 'USPS'
python MNIST-USPS/main.py --net 'bcnn' --ch 1 6 10 15 --kr 4 4 4 --norm 'bn' --lr 0.005 --meta 0.0 --epochs-per-task 240 --task-sequence 'MNIST'
```

#### BNN learning CIFAR-10 and CIFAR-100 features

```
#Fig. 4(e-f)
python main.py --net 'bnn' --norm 'bn' --decay 1e-8 --in-size 512 --hidden-layers 2048 --out-size 10 --lr 0.005 --meta 0.0 --epochs-per-task 10 --task-sequence 'vehicles' 'animals'
python main.py --net 'bnn' --norm 'bn' --decay 1e-8 --in-size 512 --hidden-layers 2048 --out-size 10 --lr 0.005 --meta 1.8 --epochs-per-task 10 --task-sequence 'vehicles' 'animals'
```

```
#Fig. 4(g-h)
python main.py --net 'bnn' --norm 'in' --scenario 'domain' --in-size 512 --hidden-layers 2048 2048 --out-size 100 --lr 0.005 --meta 0.0 --epochs-per-task 20 --task-sequence 'cifar100-2'
python main.py --net 'bnn' --norm 'in' --scenario 'domain' --in-size 512 --hidden-layers 2048 2048 --out-size 100 --lr 0.005 --meta 1.3 --epochs-per-task 20 --task-sequence 'cifar100-2'
```

#### Switching the sign of binary weights in a BNN  

This run needs approx 1 hour. Lines at the end of main.py need to be uncommented to produce the data of Fig. 5(c), (d), (e).  
```
python main.py --net 'bnn' --hidden-layers 1024 1024 --decay 1e-7 --meta 1.35 --lr 0.005 --epochs-per-task 40 --task-sequence MNIST
```

## Stream Learning (Fashion MNIST)  

The data of Fig. 4(a) is produced by the following runs (~10 hours for `--nb-subset 60`, ~10 min for `--nb-subset 1`):  
```
python main.py --task FMNIST --hidden-layers 1024 1024 --lr 0.005 --decay 1e-7 --meta 2.5 --epochs-per-task 20 --nb-subset 1  
python main.py --task FMNIST --hidden-layers 1024 1024 --lr 0.005 --decay 1e-7 --meta 0.0 --epochs-per-task 20 --nb-subset 1  
python main.py --task FMNIST --hidden-layers 1024 1024 --lr 0.005 --decay 1e-7 --meta 2.5 --epochs-per-task 20 --nb-subset 60  
python main.py --task FMNIST --hidden-layers 1024 1024 --lr 0.005 --decay 1e-7 --meta 0.0 --epochs-per-task 20 --nb-subset 60  
```

## Stream Learning (CIFAR-10)  

The data of Fig. 4(b) is produced by the following runs (~2 hours for `--nb-subset 1`, ~24 hours for `--nb-subset 20`):  

```
python main.py --lr 0.0001 --mbs 64 --source keras --nb-subset 1 --epochs-per-task 200 --meta 0.0 --init gauss --init-width 0.007  
python main.py --lr 0.0001 --mbs 64 --source keras --nb-subset 20 --epochs-per-task 200 --meta 0.0 --init gauss --init-width 0.007  
python main.py --lr 0.0001 --mbs 64 --source keras --nb-subset 1 --epochs-per-task 200 --meta 13.0 --init gauss --init-width 0.007  
python main.py --lr 0.0001 --mbs 64 --source keras --nb-subset 20 --epochs-per-task 200 --meta 13.0 --init gauss --init-width 0.007  
```

## Quadratic Binary Task  

The Quadratic Binary Problem is a jupyter notebook with Fig. 5(a), (b) already produced inside.

# License

This Repository is under the _MIT License_
