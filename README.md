# Provably Adversarially Robust Nearest Prototype Classifiers
code for our ICML 2022 paper "Provably Adversarially Robust Nearest Prototype Classifiers" (under construction)

Before running the experiments, install the dependencies with:

    pip install -r requirements.txt
    
To replicate training for MNIST, $\ell_2$ robust radius $1.58$, run the default parameters:

    python main.py 
    
To replicate training for LPIPS distance cifar, run:

    python main.py --dataset cifar10lpips --ppc 50 --decay 0.5 --bs 5000 --lr 100 --epochs 10
