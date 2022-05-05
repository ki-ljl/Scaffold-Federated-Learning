![](https://img.shields.io/badge/SCAFFOLD-Federated%20Learning-red)
# Scaffold-Federated-Learning
PyTorch implementation of SCAFFOLD (Stochastic Controlled Averaging for Federated Learning, ICML 2020).

# Environment
numpy==1.18.5

pytorch==1.10.1+cu111

# Experimental parameter settings

communication rounds: r=10,

number of local update steps: E=10,

![](http://latex.codecogs.com/svg.latex?\eta_l)=0.01,

![](http://latex.codecogs.com/svg.latex?\eta_g)=1,

total number of clients: K=10,

sampled num: |S|=5.

# Usage
```
python main.py
``` 
