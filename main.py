import jax.numpy as np
from jax import jit
from jax.lax import cond, fori_loop, scan
from functools import partial
from time import time 
import numpy as onp
from matplotlib import pyplot as plt 
import torch 
from torchvision.transforms import functional as tvF
from jax.experimental.optimizers import sgd, adam, nesterov
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
import torchvision
import warnings
import argparse
from scipy.optimize import linprog
import scipy.io as sio




def augment(x,p):
    r = torch.rand(1).item()
    if(r < p+(1-p)/5):
        return x[0,1:-1, 1:-1]
    if(r < p+2*(1-p)/5):
        return x[0,0:-2, 1:-1]
    if(r < p+3*(1-p)/5):
        return x[0,2:, 1:-1]
    if(r < p+4*(1-p)/5):
        return x[0,1:-1, 0:-2]
    else:
        return x[0,1:-1, 2:]


def get_dataset(name, ppc = None, kmeans = False, fname = None, transform = None, batch_size = 1000,aug_prob=0.7):
    def get_prototypes(loader , ppc, kmeans, fname):
        X,Y = map(np.array, next(iter(loader)))

        if(fname is not None ):
            prototypes = np.load(fname)
            assert onp.prod(X.shape[1:]) == prototypes.shape[-1]
        else:
            prototypes = []
            for i in range(10):
                indices = Y == i 
                xs = X[indices]
                xs = xs.reshape(xs.shape[0], -1)
                if(kmeans):
                    prototypes.append(KMeans(n_clusters=ppc, max_iter = 100, n_init=1).fit(xs).cluster_centers_)
                else:
                    prototypes.append(xs[onp.random.choice(xs.shape[0], ppc, replace=False), :])
            
            prototypes = np.stack(prototypes).reshape(10*ppc, *X.shape[1:])
        return prototypes 

    if(name == 'mnist'):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: augment(x,aug_prob)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:,1:-1, 1:-1]),
        ])



        DS = torchvision.datasets.MNIST
    elif(name == 'cifar10'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=2, fill=128),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        DS = torchvision.datasets.CIFAR10
    else:
        raise NotImplemented('Expected "mnist" or "cifar10", got ' + name)


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if(transform is not None):
            transform_train = transform
        trainset = DS(root="./data", train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = DS(root="./data", train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    if(ppc is not None or fname is not None):
        loader = torch.utils.data.DataLoader(DS(root="./data", train=True, download=True, 
                                transform=transform_test),  batch_size=10000, shuffle=True )
        prototypes = get_prototypes(loader, ppc, kmeans, fname)
    else:
        prototypes = None 
    return trainloader, testloader, prototypes 
               

@jit 
def _l1(Xs,y):
    return Xs, np.abs(Xs-y).sum(-1)

@jit
def scan_l1(Xs, Ys):
    return scan(_l1, Ys, Xs)[1][:,0,:]

@jit 
def _linf(Xs,y):
    return Xs, np.abs(Xs-y).max(-1)

@jit
def scan_linf(Xs, Ys):
    return scan(_linf, Ys, Xs)[1][:,0,:]


@partial(jit, static_argnums=(2,))
def pairwise_distances(Xs, Ys, pnorm):
    if(pnorm == 1):
        Xs = Xs[:,None,:]
        Ys = Ys[None,:,:]
        return scan_l1(Xs,Ys)
    elif(pnorm == np.inf):
        Xs = Xs[:,None,:]
        Ys = Ys[None,:,:]
        return scan_linf(Xs,Ys)
    elif(pnorm == 2):
        XX = (Xs**2).sum(-1).reshape(-1,1) 
        YY = (Ys**2).sum(-1).reshape(1,-1)
        return np.abs(XX + YY - 2* Xs @ Ys.T)**0.5

@jit
def _else_branch(vals):
    best_i, best_j, best_lb, lbs, idx, WX, Wdist,y, curr_i, masks = vals
    dists = (WX[:,idx] - WX[curr_i, idx]) / Wdist[curr_i]
    dists = np.where(masks[y,1], dists, np.inf)
    j = np.argmin(dists)
    return cond(dists[j] < best_lb,
            lambda _: (best_i, best_j, best_lb, lbs, idx, WX, Wdist,y,masks),
            lambda _: (curr_i, j, dists[j], lbs, idx, WX, Wdist, y, masks),
            None 
    )
@jit
def _forifun2(curr_i, carry):
    best_i, best_j, best_lb, lbs, idx, WX, Wdist,y, masks = carry
    
    return cond(lbs[curr_i] < best_lb, 
            lambda _: (best_i, best_j, best_lb, lbs, idx, WX, Wdist, y, masks),
            _else_branch,
            (best_i, best_j, best_lb, lbs, idx, WX, Wdist,y, curr_i, masks)
        )    
@jit 
def _forifun(idx, val):
    lbs_idx, ret, retidx, mindistj, minWdist, maxWdist, WX, Wdist, start, ppc,y,masks = val 
    lbs2  = (mindistj[idx] - WX[:, idx])/minWdist
    lbs2_ = (mindistj[idx] - WX[:, idx])/maxWdist 
    lbs2 = np.maximum(lbs2, lbs2_)

    best_i, best_j, best_lb, lbs, idx, *_ = fori_loop( start, start+ppc, _forifun2,  (-1, -1, -np.inf, lbs2, idx, WX, Wdist,y, masks))
    ret = ret.at[retidx, :6].set((best_i, best_j, WX[best_i, idx], WX[best_j, idx], Wdist[best_i, best_j], best_lb))
    return lbs_idx, ret, retidx+1, mindistj, minWdist, maxWdist, WX, Wdist, start, ppc,y, masks

def lb_fn(X,Y,W,pnorm,masks,ppc):
    num_classes = W.shape[0]//ppc
    WW = (W*W).sum(-1)
    XX = (X*X).sum(-1)
    _WX = WW.reshape(-1,1) - 2*W @ X.T + XX.reshape(1,-1)
    Wdist = 2*pairwise_distances(W, W, pnorm)

    minWdist = np.min(Wdist, axis=0)
    maxWdist = np.max(Wdist, axis=0)
    retidx = 0
    ret = np.zeros((X.shape[0],6 + X.shape[1]))
    for y in range(num_classes):
        xs = X[Y == y]
        if(xs.shape[0] == 0):
            continue
        WX = _WX[:, Y==y]
        rng = np.arange(xs.shape[0])
        start = y*ppc 
        arr1 = WX[masks[y,0]]
        i  = np.argmin(arr1, axis=0)
        mindisti = arr1[i,rng]

        arr2 =  WX[masks[y,1]]
        j = np.argmin(arr2, axis=0)
        mindistj = arr2[j, rng]
        i = i+start 
        j = np.where(j < start, j, j+ppc)

        ret = ret.at[retidx:retidx+xs.shape[0], 6:].set(xs)

        lbs_idxs = (mindistj - mindisti)/Wdist[i,j]
        _, ret, retidx, *_ = fori_loop(0, xs.shape[0], _forifun, (lbs_idxs, ret, retidx, mindistj, minWdist, maxWdist, WX, Wdist, start, ppc, y, masks))
    return ret 

        
def grad_lb(W,X,I,J, dIJ,lbs,train_eps):
    gradI = -(W[I]-X)/dIJ - (lbs) * (W[I] - W[J])/dIJ**2
    gradJ =  (W[J]-X)/dIJ + (lbs) * (W[I] - W[J])/dIJ**2

    gradI = np.where(lbs.T < train_eps, gradI.T, 0).T
    gradJ = np.where(lbs.T < train_eps, gradJ.T, 0).T

    gradW = np.zeros_like(W)
    gradW = gradW.at[I].add(gradI)
    gradW = gradW.at[J].add(gradJ)

    return -gradW/X.shape[0]



def train(train_loader, test_loader, W, epochs, pnorm, masks, ppc, train_eps, test_eps, lr,decay,ftrs,aug_prob,momentum):
    lrfun = lambda i: lr * (decay ** (i) )
    opt_init, opt_update, get_params=nesterov(lrfun, momentum)
    opt_state = opt_init(W)

    cur_max = 0
    for e in range(1,1+epochs):
        start = time()
        loss = 0
        L = 0
        for bidx, (X, Y) in enumerate(train_loader):
            print(X.shape, ftrs(X).shape)
            X = ftrs(X)
            Y = np.array(Y) 

            W = get_params(opt_state)
            lowerbounds = lb_fn(X,Y,W,pnorm,masks,ppc)
            I, J, dIX2, dJX2, dIJ, lbs =  (lowerbounds.T)[:6]

            I = I.astype(np.int32); J =J.astype(np.int32)
            dIJ = dIJ[:, None]/2; lbs = lbs[:,None]; X = lowerbounds[:,6:]
            gradW = grad_lb(W,X,I,J,dIJ,lbs,train_eps)

            correctly_classified = np.sum(lbs[:,0] > 0)
            robustly_classified  = np.sum(lbs[:,0] > test_eps)

            opt_state = opt_update(e, gradW, opt_state)
            loss = np.sum(np.maximum(train_eps-lbs, 0))

            L += loss
            print(f'epoch: {e:3d}, batch: {bidx:3d}, RA: {robustly_classified/X.shape[0]:.4f}, CA: {correctly_classified/X.shape[0]:.4f}, loss: {loss:.5f}, lr: { (lrfun(e) ):.4f}, grad norm: {np.linalg.norm(gradW)}')
        correct = 0
        total = 0
        correctly_classified = 0
        robustly_classified = 0
        T = time()
        for bidx, (X, Y) in enumerate(test_loader):
            X = ftrs(X)
            Y = np.array(Y) 

            lowerbounds = lb_fn(X,Y,W,pnorm,masks,ppc)
            lbs =  (lowerbounds.T)[5]
            correctly_classified += np.sum(lbs > 0)
            robustly_classified  += np.sum(lbs > test_eps)
            total += X.shape[0] 

        print(f'epoch: {e}, RA: {robustly_classified/total:.4f}, CA: {correctly_classified/total:.4f}, loss: {L:.5f}, epoch took {time()-start:.2f} sec')
        np.save('ppc: ' + str(ppc) + " " + 'bs: ' + str(X.shape[0]) + " " + 'lr: '+ str(lr) + " " + 'decay: ' + str(decay) + " " + 'train eps:' + str(train_eps) + " " + 'aug prob:'+ str(aug_prob) + " " + 'momentum: '+ str(momentum), onp.array(W))
    return W, lbs

def get_masks(ppc, num_classes):
    masks = []
    for i in range(num_classes):
        start =  i    * ppc 
        end   = (i+1) * ppc
            
        rng = np.arange(ppc*num_classes)
        mask1 = (rng >= start) & (rng < end)
        mask2 = ~mask1
        masks.append((mask1, mask2))
    return np.array(masks)    


if(__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppc', type=int, required=False, default = 400)
    parser.add_argument('--dataset', type=str, required=False, default = 'mnist')
    parser.add_argument('--decay', type=float,required=False,default=0.96)
    parser.add_argument('--bs', type=int, required=False,default=500)
    parser.add_argument('--lr', type=float,required=False,default=40)
    parser.add_argument('--epochs', type=int,required=False,default=100)
    parser.add_argument('--eps',type=float,required=False,default=2.5)
    parser.add_argument('--aug_prob',type=float,required=False,default=0.5)
    parser.add_argument('--momentum',type=float,required=False,default=0.9)
    decay   = parser.parse_args().decay
    ppc     = parser.parse_args().ppc
    bs      = parser.parse_args().bs
    lr      = parser.parse_args().lr
    dataset = parser.parse_args().dataset
    epochs  = parser.parse_args().epochs
    aug_prob= parser.parse_args().aug_prob
    momentum = parser.parse_args().momentum

    if(dataset  == 'mnist'):
        test_eps = 1.58
        train_eps = 2.5
        train_eps = parser.parse_args().eps
        ftrs = lambda X : np.array(X.reshape(X.shape[0], -1))
    elif(dataset == 'cifar10'):
        test_eps = 36/255
        train_eps = 4
        ftrs = lambda X: np.array(X.reshape(X.shape[0], -1))
    elif(dataset == 'cifar10lpips'):
        test_eps = 0.5
        train_eps = 1
        from perceptual_advex.distances import LPIPSDistance, normalize_flatten_features
        from perceptual_advex.perceptual_attacks import get_lpips_model

        lpips_model = get_lpips_model('alexnet_cifar')
        lpips_distance = LPIPSDistance(lpips_model, include_image_as_activation=False)

        def ftrs(X):
            with(torch.no_grad()):
                print(X.shape)
                if(type(X) is not torch.Tensor):
                    X = torch.tensor(onp.array(X))
                return np.array(normalize_flatten_features(lpips_distance.features(X)).detach().numpy())
    else:
        raise NotImplementedError

    num_classes = 10

    trn_loader, tst_loader, W = get_dataset(dataset[:7], ppc = ppc, batch_size=bs,aug_prob=aug_prob)
    masks = get_masks(ppc, num_classes)
    W = ftrs(W).reshape(ppc*num_classes, -1)
    print(W.shape)


    W, ret  = train(trn_loader, tst_loader, W, epochs = epochs, pnorm = 2, masks=masks, ppc=ppc, train_eps = train_eps, test_eps = test_eps, lr=lr,decay=decay, ftrs=ftrs, aug_prob=aug_prob, momentum=momentum)
