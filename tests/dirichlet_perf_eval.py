import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import gmean, kde
import scipy.stats as st

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold

from scipy.optimize import fmin, newton, minimize
from scipy.optimize import Bounds

import smote_cd
import dirichlet_regression

import time

from multiprocessing import Pool

def f_spatial(x, X, Y, Z, W, epsilon=0):
    K = X.shape[-1]
    J = Y.shape[-1]
    n = X.shape[0]
    beta = x[:K*J].reshape((K,J))
    beta[:,0] = 0
    gamma_var = x[K*J:-1]
    rho = x[-1]
    Minv = np.linalg.inv(np.identity(n) - rho*W)
    mu = dirichlet_regression.compute_mu_spatial_2(X, beta, rho=rho, W=W, Minv=Minv)
    phi = np.exp(np.matmul(Z,gamma_var))
    return -dirichlet_regression.dirichlet_loglikelihood(mu,phi,Y,epsilon=epsilon)

def fprime_spatial(x, X, Y, Z, W, epsilon=0):
    K = X.shape[-1]
    J = Y.shape[-1]
    n = X.shape[0]
    beta = x[:K*J].reshape((K,J))
    beta[:,0] = 0
    gamma_var = x[K*J:-1]
    rho = x[-1]
    Minv = np.linalg.inv(np.identity(n) - rho*W)
    MinvX = np.matmul(Minv,X)
    mu = dirichlet_regression.compute_mu_spatial_2(X, beta, rho=rho, W=W, MinvX=MinvX)
    phi = np.exp(np.matmul(Z,gamma_var))

    beta_grad = dirichlet_regression.dirichlet_gradient_wrt_beta(mu, phi, MinvX, Y, epsilon=epsilon)
    beta_grad[:,0] = 0
    gamma_grad = dirichlet_regression.dirichlet_derivative_wrt_gamma(mu, phi, beta, MinvX, Y, Z, epsilon=epsilon)
    rho_derivative = dirichlet_regression.dirichlet_derivative_wrt_rho(mu, phi, Minv, beta, W, X, Y, Z, MinvX, epsilon=epsilon)
    return(-np.concatenate([beta_grad.flatten(),gamma_grad,[rho_derivative]]))

def f_no_spatial(x, X, Y, Z, epsilon=0):
    K = X.shape[-1]
    J = Y.shape[-1]
    beta = x[:K*J].reshape((K,J))
    beta[:,0] = 0
    gamma_var = x[K*J:]
    mu = dirichlet_regression.compute_mu_3(X, beta)
    phi = np.exp(np.matmul(Z,gamma_var))
    return -dirichlet_regression.dirichlet_loglikelihood(mu,phi,Y,epsilon=epsilon)

def fprime_no_spatial(x, X, Y, Z, epsilon=0):
    K = X.shape[-1]
    J = Y.shape[-1]
    beta = x[:K*J].reshape((K,J))
    beta[:,0] = 0
    gamma_var = x[K*J:]
    mu = dirichlet_regression.compute_mu_3(X, beta)
    phi = np.exp(np.matmul(Z,gamma_var))
    beta_grad = dirichlet_regression.dirichlet_gradient_wrt_beta(mu, phi, X, Y, epsilon=epsilon)
    beta_grad[:,0] = 0
    gamma_grad = dirichlet_regression.dirichlet_derivative_wrt_gamma(mu, phi, beta, X, Y, Z, epsilon=epsilon)
    return(-np.concatenate([beta_grad.flatten(),gamma_grad]))


#For Maupiti:
#n_features = 16
#n_classes = 4
# For elections:
n_features = 25
n_classes = 3

beta = np.zeros((1+n_features)*n_classes+(1+n_features))[:(1+n_features)*n_classes].reshape(((1+n_features),n_classes))
beta[:,0] = 0

gamma = np.zeros(n_features)
rho = [0.6]

params0 = np.concatenate([beta.flatten(),gamma,rho])
params0_ns = np.concatenate([beta.flatten(),gamma])

min_bounds = -np.inf*np.ones(len(params0)) 
max_bounds = np.inf*np.ones(len(params0))
min_bounds[-1] = -1
max_bounds[-1] = 1
bounds = Bounds(min_bounds, max_bounds)

def eval_perf_maupiti(random_s, X, Y, Z, W, k_folds=4):
    
    r2_ns, r2_s = [], []
    rmse_ns, rmse_s = [], []
    aic_ns, aic_s = [], []
    crossentropy_ns, crossentropy_s = [], []
    
    r2_test_ns, r2_test_s = [], []
    rmse_test_ns, rmse_test_s = [], []
    aic_test_ns, aic_test_s = [], []
    crossentropy_test_ns, crossentropy_test_s = [], []
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_s)
    kf.get_n_splits(X)
    
    for train_ind, test_ind in kf.split(X):
        X_train, X_test = X[train_ind], X[test_ind]
        Y_train, Y_test = Y[train_ind], Y[test_ind]
        Z_train, Z_test = Z[train_ind], Z[test_ind]
        W_train, W_test = W[train_ind][:,train_ind], W[test_ind][:,test_ind]
        
        solution_ns = minimize(f_no_spatial, params0_ns, args=(X_train, Y_train, Z_train), jac=fprime_no_spatial)
        
        beta_1_ns = solution_ns.x[:(n_features+1)*n_classes].reshape((n_features+1,n_classes))
        mu_1_ns = dirichlet_regression.compute_mu_2(X, beta_1_ns)
        gamma_var_1_ns = solution_ns.x[(n_features+1)*n_classes:]
        phi_1_ns = np.exp(np.matmul(Z,gamma_var_1_ns))
        
        r2_ns.append(r2_score(Y,mu_1_ns))
        rmse_ns.append(mean_squared_error(Y,mu_1_ns,squared=False))
        aic_ns.append(2*len(solution_ns.x) - 2*dirichlet_regression.dirichlet_loglikelihood(mu_1_ns,phi_1_ns,Y))
        crossentropy_ns.append(np.sum(Y*np.log(mu_1_ns)))
        
        r2_test_ns.append(r2_score(Y_test,mu_1_ns[test_ind]))
        rmse_test_ns.append(mean_squared_error(Y_test,mu_1_ns[test_ind],squared=False))
        aic_test_ns.append(2*len(solution_ns.x) - 2*dirichlet_regression.dirichlet_loglikelihood(mu_1_ns[test_ind], phi_1_ns[test_ind], Y_test))
        crossentropy_test_ns.append(np.sum(Y_test*np.log(mu_1_ns[test_ind])))
        
        solution_s = minimize(f_spatial, params0, args=(X_train, Y_train, Z_train, W_train), bounds=bounds, jac=fprime_spatial)
        
        beta_1_s = solution_s.x[:(n_features+1)*n_classes].reshape((n_features+1,n_classes))
        rho_1_s = solution_s.x[-1]

        mu_1_s = dirichlet_regression.compute_mu_spatial_2(X, beta_1_s, rho_1_s, W)

        gamma_var_1_s = solution_s.x[(n_features+1)*n_classes:-1]
        phi_1_s = np.exp(np.matmul(Z,gamma_var_1_s))
        
        r2_s.append(r2_score(Y,mu_1_s))
        rmse_s.append(mean_squared_error(Y,mu_1_s,squared=False))
        aic_s.append(2*len(solution_s.x) - 2*dirichlet_regression.dirichlet_loglikelihood(mu_1_s,phi_1_s,Y))
        crossentropy_s.append(np.sum(Y*np.log(mu_1_s)))
        
        r2_test_s.append(r2_score(Y_test,mu_1_s[test_ind]))
        rmse_test_s.append(mean_squared_error(Y_test,mu_1_s[test_ind],squared=False))
        aic_test_s.append(2*len(solution_s.x) - 2*dirichlet_regression.dirichlet_loglikelihood(mu_1_s[test_ind], phi_1_s[test_ind], Y_test))
        crossentropy_test_s.append(np.sum(Y_test*np.log(mu_1_s[test_ind])))
        
    return(r2_ns, r2_s, rmse_ns, rmse_s, aic_ns, aic_s, crossentropy_ns, crossentropy_s, 
           r2_test_ns, r2_test_s, rmse_test_ns, rmse_test_s, aic_test_ns, aic_test_s, crossentropy_test_ns, crossentropy_test_s)

def eval_perf(random_s, X, Y, Z, W, k_folds=4):
    
    r2_ns, r2_s = [], []
    rmse_ns, rmse_s = [], []
    aic_ns, aic_s = [], []
    crossentropy_ns, crossentropy_s = [], []
    similarity_ns, similarity_s = [], [] 
    
    r2_test_ns, r2_test_s = [], []
    rmse_test_ns, rmse_test_s = [], []
    aic_test_ns, aic_test_s = [], []
    crossentropy_test_ns, crossentropy_test_s = [], []
    similarity_test_ns, similarity_test_s = [], []
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_s)
    kf.get_n_splits(X)
    
    for train_ind, test_ind in kf.split(X):
        X_train, X_test = X[train_ind], X[test_ind]
        Y_train, Y_test = Y[train_ind], Y[test_ind]
        Z_train, Z_test = Z[train_ind], Z[test_ind]
        W_train, W_test = W[train_ind][:,train_ind], W[test_ind][:,test_ind]
        
        solution_ns = minimize(f_no_spatial, params0_ns, args=(X_train, Y_train, Z_train), jac=fprime_no_spatial)
        
        beta_1_ns = solution_ns.x[:(n_features+1)*n_classes].reshape((n_features+1,n_classes))
        mu_1_ns = dirichlet_regression.compute_mu_2(X, beta_1_ns)
        gamma_var_1_ns = solution_ns.x[(n_features+1)*n_classes:]
        phi_1_ns = np.exp(np.matmul(Z,gamma_var_1_ns))
        
        r2_ns.append(r2_score(Y,mu_1_ns))
        rmse_ns.append(mean_squared_error(Y,mu_1_ns,squared=False))
        aic_ns.append(2*len(solution_ns.x) - 2*dirichlet_regression.dirichlet_loglikelihood(mu_1_ns,phi_1_ns,Y))
        crossentropy_ns.append(np.sum(Y*np.log(mu_1_ns)))
        similarity_ns.append(np.mean([np.dot(Y[i],mu_1_ns[i]) / (np.linalg.norm(Y[i])*np.linalg.norm(mu_1_ns[i])) for i in range(len(Y))]))
        
        r2_test_ns.append(r2_score(Y_test,mu_1_ns[test_ind]))
        rmse_test_ns.append(mean_squared_error(Y_test,mu_1_ns[test_ind],squared=False))
        aic_test_ns.append(2*len(solution_ns.x) - 2*dirichlet_regression.dirichlet_loglikelihood(mu_1_ns[test_ind], phi_1_ns[test_ind], Y_test))
        crossentropy_test_ns.append(np.sum(Y_test*np.log(mu_1_ns[test_ind])))
        similarity_test_ns.append(np.mean([np.dot(Y_test[i],mu_1_ns[test_ind][i]) / (np.linalg.norm(Y_test[i])*np.linalg.norm(mu_1_ns[test_ind][i])) for i in range(len(Y_test))]))
        
        solution_s = minimize(f_spatial, params0, args=(X_train, Y_train, Z_train, W_train), bounds=bounds, jac=fprime_spatial)
        
        beta_1_s = solution_s.x[:(n_features+1)*n_classes].reshape((n_features+1,n_classes))
        rho_1_s = solution_s.x[-1]

        mu_1_s = dirichlet_regression.compute_mu_spatial_2(X, beta_1_s, rho_1_s, W)

        gamma_var_1_s = solution_s.x[(n_features+1)*n_classes:-1]
        phi_1_s = np.exp(np.matmul(Z,gamma_var_1_s))
        
        r2_s.append(r2_score(Y,mu_1_s))
        rmse_s.append(mean_squared_error(Y,mu_1_s,squared=False))
        aic_s.append(2*len(solution_s.x) - 2*dirichlet_regression.dirichlet_loglikelihood(mu_1_s,phi_1_s,Y))
        crossentropy_s.append(np.sum(Y*np.log(mu_1_s)))
        similarity_s.append(np.mean([np.dot(Y[i],mu_1_s[i]) / (np.linalg.norm(Y[i])*np.linalg.norm(mu_1_s[i])) for i in range(len(Y))]))
        
        r2_test_s.append(r2_score(Y_test,mu_1_s[test_ind]))
        rmse_test_s.append(mean_squared_error(Y_test,mu_1_s[test_ind],squared=False))
        aic_test_s.append(2*len(solution_s.x) - 2*dirichlet_regression.dirichlet_loglikelihood(mu_1_s[test_ind], phi_1_s[test_ind], Y_test))
        crossentropy_test_s.append(np.sum(Y_test*np.log(mu_1_s[test_ind])))
        similarity_test_s.append(np.mean([np.dot(Y_test[i],mu_1_s[test_ind][i]) / (np.linalg.norm(Y_test[i])*np.linalg.norm(mu_1_s[test_ind][i])) for i in range(len(Y_test))]))
        
    return(r2_ns, r2_s, rmse_ns, rmse_s, aic_ns, aic_s, crossentropy_ns, crossentropy_s, similarity_ns, similarity_s,
           r2_test_ns, r2_test_s, rmse_test_ns, rmse_test_s, aic_test_ns, aic_test_s, crossentropy_test_ns, crossentropy_test_s,
          similarity_test_ns, similarity_test_s)