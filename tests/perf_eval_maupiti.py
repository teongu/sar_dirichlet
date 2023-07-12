import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import gmean, kde
import scipy.stats as st

from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold

import smote_cd
import dirichlet_regression

import time

from multiprocessing import Pool


def cos_similarity(x1,x2):
    return(np.mean([np.dot(x1[i],x2[i])/(np.linalg.norm(x1[i])*np.linalg.norm(x2[i])) for i in range(len(x1))]))

#For Maupiti:
n_features = 16
n_classes = 4

gamma = np.zeros(n_features)
rho = 0.8

def main_eval(random_s, X, Y, Z, W, k_folds=4):
    
    n_samples = X.shape[0]
    
    r2_ns, r2_s = [], []
    rmse_ns, rmse_s = [], []
    aic_ns, aic_s = [], []
    crossentropy_ns, crossentropy_s = [], []
    sim_ns, sim_s = [], []
    
    r2_test_ns, r2_test_s = [], []
    rmse_test_ns, rmse_test_s = [], []
    aic_test_ns, aic_test_s = [], []
    crossentropy_test_ns, crossentropy_test_s = [], []
    sim_test_ns, sim_test_s = [], []
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_s)
    kf.get_n_splits(X)
    
    for train_ind, test_ind in kf.split(X):
        X_train, X_test = X[train_ind], X[test_ind]
        Y_train, Y_test = Y[train_ind], Y[test_ind]
        Z_train, Z_test = Z[train_ind], Z[test_ind]
        W_train, W_test = W[train_ind][:,train_ind], W[test_ind][:,test_ind]
        
        dirichRegressor_ns = dirichlet_regression.dirichletRegressor()
        dirichRegressor_ns.fit(X_train, Y_train, parametrization='alternative', gamma_0=gamma, Z=Z_train)
        
        r2_ns.append(r2_score(Y_train,dirichRegressor_ns.mu))
        rmse_ns.append(mean_squared_error(Y_train,dirichRegressor_ns.mu,squared=False))
        crossentropy_ns.append(1/n_samples*np.sum(Y_train*np.log(dirichRegressor_ns.mu)))
        aic_ns.append(-2*dirichlet_regression.dirichlet_loglikelihood(dirichRegressor_ns.mu,dirichRegressor_ns.phi,Y_train)+2*67)
        sim_ns.append(cos_similarity(Y_train,dirichRegressor_ns.mu))
        
        beta_1_ns = np.ones((n_features+1,n_classes))
        beta_1_ns[:,1:] = np.copy(dirichRegressor_ns.beta)
        X_1 = np.ones((n_samples,n_features+1))
        X_1[:,1:] = np.copy(X)
        mu_1_ns = dirichlet_regression.compute_mu(X_1, beta_1_ns)
        gamma_var_1_ns = dirichRegressor_ns.gamma
        phi_1_ns = np.exp(np.matmul(Z,gamma_var_1_ns))
        
        r2_test_ns.append(r2_score(Y_test,mu_1_ns[test_ind]))
        rmse_test_ns.append(mean_squared_error(Y_test,mu_1_ns[test_ind],squared=False))
        crossentropy_test_ns.append(1/n_samples*np.sum(Y_test*np.log(mu_1_ns[test_ind])))
        sim_test_ns.append(cos_similarity(Y_test,mu_1_ns[test_ind]))
        
        
        dirichRegressor_s = dirichlet_regression.dirichletRegressor(spatial=True,maxfun=10000)
        dirichRegressor_s.fit(X_train, Y_train, W=W_train, parametrization='alternative', rho_0=rho, gamma_0=gamma, Z=Z_train)
        
        r2_s.append(r2_score(Y_train,dirichRegressor_s.mu))
        rmse_s.append(mean_squared_error(Y_train,dirichRegressor_s.mu,squared=False))
        crossentropy_s.append(1/n_samples*np.sum(Y_train*np.log(dirichRegressor_s.mu)))
        aic_s.append(-2*dirichlet_regression.dirichlet_loglikelihood(dirichRegressor_s.mu,dirichRegressor_s.phi,Y_train)+2*68)
        sim_s.append(cos_similarity(Y_train,dirichRegressor_s.mu))
        
        M = np.identity(n_samples) - dirichRegressor_s.rho * W
        beta_1_s = np.ones((n_features+1,n_classes))
        beta_1_s[:,1:] = np.copy(dirichRegressor_s.beta)
        mu_1_s = dirichlet_regression.compute_mu_spatial(X_1, beta_1_s, M)
        gamma_var_1_s = dirichRegressor_s.gamma
        phi_1_s = np.exp(np.matmul(Z,gamma_var_1_s))
        
        r2_test_s.append(r2_score(Y_test,mu_1_s[test_ind]))
        rmse_test_s.append(mean_squared_error(Y_test,mu_1_s[test_ind],squared=False))
        crossentropy_test_s.append(1/n_samples*np.sum(Y_test*np.log(mu_1_s[test_ind])))
        sim_test_s.append(cos_similarity(Y_test,mu_1_s[test_ind]))
        
    return(r2_ns, r2_s, rmse_ns, rmse_s, aic_ns, aic_s, crossentropy_ns, crossentropy_s, sim_ns, sim_s,
           r2_test_ns, r2_test_s, rmse_test_ns, rmse_test_s, aic_test_ns, aic_test_s, crossentropy_test_ns, crossentropy_test_s, sim_test_ns, sim_test_s)
