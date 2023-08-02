import numpy as np
import pandas as pd
from smote_cd.dataset_generation import softmax
from scipy.special import gamma, digamma, polygamma, loggamma
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

import time #to be removed

def compute_mu(X, beta):
    n = np.shape(X)[0]
    J = np.shape(beta)[1]
    Xbeta = np.matmul(X,beta)
    mu = np.zeros((n,J))
    # compute the softmax
    exp_Xbeta = np.exp(Xbeta)
    sum_exp_Xbeta = np.sum(exp_Xbeta,axis=1)
    mu = exp_Xbeta/sum_exp_Xbeta[:,None]
    return mu

def compute_mu_2(X, beta):
    n = np.shape(X)[0]
    J = np.shape(beta)[1]
    Xbeta = np.matmul(X,beta)
    mu = np.zeros((n,J))
    # compute the softmax
    for i in range(n):
        for j in range(J):
            mu[i,j] = 1/np.sum(np.exp(Xbeta[i]-Xbeta[i,j]))
    return mu

def compute_mu_3(X, beta):
    n = np.shape(X)[0]
    J = np.shape(beta)[1]
    Xbeta = np.matmul(X,beta)
    mu = np.zeros((n,J))
    # compute the softmax
    exp_Xbeta = np.exp(Xbeta)
    exp_Xbeta[Xbeta>700] = 1e305
    sum_exp_Xbeta = np.sum(exp_Xbeta,axis=1)
    mu = exp_Xbeta/sum_exp_Xbeta[:,None]
    mu[mu==0]=1e-305
    return mu

def compute_mu_spatial(X, beta, rho=None, W=None, Minv=None, MinvX=None):
    n = np.shape(X)[0]
    J = np.shape(beta)[1]
        
    if MinvX is None:
        if Minv is None:
            Minv = sparse.linalg.splu(sparse.csc_matrix(np.identity(n) - rho*W)).solve(np.eye(n))
            Minv = np.transpose(Minv)
        Xbeta = np.matmul(X,beta)
        MXbeta = np.matmul(Minv,Xbeta)
    else:
        MXbeta = np.matmul(MinvX,beta)
        
    mu = np.zeros((n,J))
    exp_MXbeta = np.exp(MXbeta)
    sum_exp_MXbeta = np.sum(exp_MXbeta,axis=1)
    mu = exp_MXbeta/sum_exp_MXbeta[:,None]
    return mu

def compute_mu_spatial_2(X, beta, rho=None, W=None, Minv=None, MinvX=None):
    n = np.shape(X)[0]
    J = np.shape(beta)[1]
    if MinvX is None:
        if Minv is None:
            I_n = np.identity(n)
            Minv = np.linalg.inv(I_n - rho*W)
        Xbeta = np.matmul(X,beta)
        MXbeta = np.matmul(Minv,Xbeta)
    else:
        MXbeta = np.matmul(MinvX,beta)
    mu = np.zeros((n,J))
    exp_MXbeta = np.exp(MXbeta)
    exp_MXbeta[MXbeta>700] = 1e305
    sum_exp_MXbeta = np.sum(exp_MXbeta,axis=1)
    mu = exp_MXbeta/sum_exp_MXbeta[:,None]
    mu[mu==0]=1e-305
    return mu

def compute_mu_spatial_3(X, beta, rho=None, W=None, Minv=None, MinvX=None):
    n = np.shape(X)[0]
    J = np.shape(beta)[1]
        
    if MinvX is None:
        if Minv is None:
            Minv = sparse.linalg.inv(sparse.csc_matrix(np.identity(n) - rho*W)).toarray()
        Xbeta = np.matmul(X,beta)
        MXbeta = np.matmul(Minv,Xbeta)
    else:
        MXbeta = np.matmul(MinvX,beta)
        
    mu = np.zeros((n,J))
    exp_MXbeta = np.exp(MXbeta)
    sum_exp_MXbeta = np.sum(exp_MXbeta,axis=1)
    mu = exp_MXbeta/sum_exp_MXbeta[:,None]
    return mu


def compute_mu_spatial_opti(X, beta, M, Xbeta=None, MinvX=None, MXbeta=None):
    n = np.shape(X)[0]
    J = np.shape(beta)[1]
    if MXbeta is None:
        if MinvX is None:
            if Xbeta is None:
                Xbeta = np.matmul(X,beta)
            MXbeta = sparse.linalg.spsolve(sparse.csc_matrix(M),Xbeta)
        else:
            MXbeta = np.matmul(MinvX,beta)
    mu = np.zeros((n,J))
    exp_MXbeta = np.exp(MXbeta)
    exp_MXbeta[MXbeta>700] = 1e305
    sum_exp_MXbeta = np.sum(exp_MXbeta,axis=1)
    mu = exp_MXbeta/sum_exp_MXbeta[:,None]
    mu[mu==0]=1e-305
    return mu

####################


def dirichlet_loglikelihood(mu,phi,Y,epsilon=0):
    n = np.shape(Y)[0]
    sum_ll = 0
    for i in range(n):
        phi_mu = phi[i]*mu[i]
        sum_ll += loggamma(phi[i]) - np.sum(loggamma(phi_mu)) + np.sum((phi_mu-1)*np.log(Y[i]+epsilon))
    return sum_ll



def dirichlet_gradient_wrt_beta(mu, phi, X, Y, epsilon=0):
    K = np.shape(X)[1] #nb of features
    J = np.shape(Y)[1] #nb of classes
    n = np.shape(Y)[0] 
    gradient = np.zeros((K,J))
    for i in range(n):
        sum_mu_ij = np.sum(mu[i])
        digamma_sum_mu_ij = digamma(sum_mu_ij) 
        digamma_phi_mu = digamma(phi[i]*mu[i])
        sum_mu_digamma_phi_mu = np.sum(mu[i] * digamma_phi_mu)
        logY = np.log(Y[i] + epsilon)
        sum_mu_logY = np.sum(mu[i]*logY)
        term_final = phi[i]*mu[i,:]*( sum_mu_digamma_phi_mu - sum_mu_logY + logY - digamma_phi_mu )
        #for p in range(K):
        #    gradient[p,:] += X[i,p]*term_final
        gradient += np.outer(X[i],term_final)
    return gradient


def dirichlet_derivative_wrt_gamma(mu, phi, Y, Z, epsilon=0):
    n = np.shape(Y)[0]
    derivative_gamma = np.zeros(np.shape(Z)[1])
    for i in range(n):
        derivative_gamma += Z[i,:]*phi[i]*( digamma(phi[i])+np.sum( mu[i]*(np.log(Y[i]+epsilon)-digamma(phi[i]*mu[i])) ) )
    return(derivative_gamma)

def dirichlet_hessian_wrt_gamma(mu, phi, beta, X, Y, Z, epsilon=0):
    K = np.shape(Z)[1]
    n = np.shape(Y)[0]
    hessian = np.zeros((K,K))
    for i in range(n):
        phi_mu = phi[i]*mu[i]
        logY = np.log(Y[i]+epsilon)
        temp = phi[i] * Z[i,:] * ( phi[i]*polygamma(1,phi[i]) + digamma(phi[i]) - 
                                  phi[i]*np.sum( mu[i]**2*polygamma(1,phi_mu) ) 
                                  + np.sum(mu[i]*(logY-digamma(phi_mu))) )
        for k in range(K):
            hessian[:,k] += Z[i,k]*temp
    return(hessian)

def dirichlet_hessian_wrt_gamma_2(mu, phi, beta, X, Y, Z, epsilon=0):
    K = np.shape(Z)[1]
    n = np.shape(Y)[0]
    hessian = np.zeros((K,K))
    for i in range(n):
        phi_mu = phi[i]*mu[i]
        trigamma_phi_mu = polygamma(1,phi_mu)
        trigamma_phi_mu[np.isinf(trigamma_phi_mu)] = 1e305
        logY = np.log(Y[i]+epsilon)
        temp = phi[i] * Z[i,:] * ( phi[i]*polygamma(1,phi[i]) + digamma(phi[i]) - 
                                  phi[i]*np.sum( mu[i]**2*trigamma_phi_mu ) 
                                  + np.sum(mu[i]*(logY-digamma(phi_mu))) )
        for k in range(K):
            hessian[:,k] += Z[i,k]*temp
    return(hessian)

def dirichlet_hessian_wrt_beta(mu, phi, X, Y, alpha=None, epsilon=0):
    K = np.shape(X)[1] #nb of features
    J = np.shape(Y)[1] #nb of classes
    n = np.shape(Y)[0] 
    hessian = np.zeros((K,J,K,J))
    if alpha is None:
        alpha = mu*phi[:,None]
    for i in range(n):
        digamma_alpha = digamma(alpha[i])
        trigamma_alpha = polygamma(1,alpha[i])
        
        mu_2_trigamma = mu[i]**2*trigamma_alpha
        #mu_2_trigamma[mu[i]==0] = 0
        phi_sum_mu_2_trigamma = phi[i]*np.sum(mu_2_trigamma)
        
        logY = np.log(Y[i]+epsilon)
        sum_mu_log = 2*np.sum(mu[i]*logY)
        
        mu_digamma_alpha = mu[i]*digamma_alpha
        #mu_digamma_alpha[mu[i]==0] = 0
        sum_mu_digamma = 2*np.sum(mu_digamma_alpha)
        
        mu_digamma_log = mu[i]*(digamma_alpha-logY)
        #mu_digamma_log[mu[i]==0] = 0
        sum_mu_digamma_log = np.sum(mu_digamma_log)
        for p in range(K):
            phi_Xip = phi[i]*X[i,p]
            for d in range(J):
                phi_Xip_muid = phi_Xip*mu[i,d]
                if alpha[i,d]==0:
                    alpha_times_trigamma_alpha = np.zeros(np.shape(trigamma_alpha[d]))
                else:
                    alpha_times_trigamma_alpha = alpha[i,d]*trigamma_alpha[d]
                for q in range(K):
                    phi_Xip_muid_Xiq = phi_Xip_muid*X[i,q]
                    for c in range(J):
                        if c!=d:
                            hessian[p,d,q,c] += phi_Xip_muid_Xiq*mu[i,c] * ( digamma_alpha[c] + digamma_alpha[d]
                                                                           - logY[c] - logY[d]
                                                                           + alpha[i,c]*trigamma_alpha[c] + alpha_times_trigamma_alpha
                                                                           - phi_sum_mu_2_trigamma 
                                                                           + sum_mu_log
                                                                           - sum_mu_digamma )
                        else:
                            hessian[p,d,q,c] += phi_Xip_muid_Xiq*mu[i,c] * ( 2*digamma_alpha[c] - 2*logY[c] 
                                                                           + 2*alpha_times_trigamma_alpha
                                                                           - phi_sum_mu_2_trigamma
                                                                           + sum_mu_log
                                                                           - sum_mu_digamma
                                                                           - phi[i]*trigamma_alpha[c])
                            hessian[p,d,q,c] += phi_Xip_muid_Xiq * (sum_mu_digamma_log - digamma_alpha[c] + logY[c] )
    # fill the symmetric values we didn't compute
    #for p in range(K):
    #    for d in range(J):
    #        for q in range(p):
    #            for c in range(J):
    #                hessian[p,d,q,c] = hessian[q,c,p,d]
    return hessian


def dirichlet_second_derivative_beta_gamma(mu, phi, beta, X, Y, Z, epsilon=0):
    K = np.shape(X)[1]
    J = np.shape(Y)[1]
    if len(phi.shape)==1:
        K_phi=1
    else:
        K_phi = phi.shape[-1]
    n = np.shape(Y)[0]
    derivatives = np.zeros((K_phi,K,J))
    for i in range(n):
        digamma_phi_mu = digamma(phi[i]*mu[i])
        trigamma_phi_mu = polygamma(1,phi[i]*mu[i])
        logYi = np.log(Y[i]+epsilon)
        sum_temp = np.sum(mu[i]*(logYi - digamma_phi_mu - mu[i]*phi[i]*trigamma_phi_mu))
        phi_Z_i = phi[i]*Z[i,:]
        for d in range(J):
            temp_d = mu[i,d]*( logYi[d] - digamma_phi_mu[d] - mu[i,d]*phi[i]*trigamma_phi_mu[d] - sum_temp )
            for p in range(K):
                derivatives[:,p,d] += phi_Z_i*X[i,p]*temp_d
    return(derivatives)

def dirichlet_second_derivative_beta_gamma_2(mu, phi, beta, X, Y, Z, epsilon=0):
    K = np.shape(X)[1]
    J = np.shape(Y)[1]
    if len(phi.shape)==1:
        K_phi=1
    else:
        K_phi = phi.shape[-1]
    n = np.shape(Y)[0]
    derivatives = np.zeros((K_phi,K,J))
    for i in range(n):
        digamma_phi_mu = digamma(phi[i]*mu[i])
        trigamma_phi_mu = polygamma(1,phi[i]*mu[i])
        trigamma_phi_mu[np.isinf(trigamma_phi_mu)] = 1e305
        logYi = np.log(Y[i]+epsilon)
        sum_temp = np.sum(mu[i]*(logYi - digamma_phi_mu - mu[i]*phi[i]*trigamma_phi_mu))
        phi_Z_i = phi[i]*Z[i,:]
        for d in range(J):
            temp_d = mu[i,d]*( logYi[d] - digamma_phi_mu[d] - mu[i,d]*phi[i]*trigamma_phi_mu[d] - sum_temp )
            for p in range(K):
                derivatives[:,p,d] += phi_Z_i*X[i,p]*temp_d
    return(derivatives)

def dirichlet_derivative_wrt_rho(mu, phi, beta, M, W, X, Y, Z, MinvX=None, MinvXbeta=None, alpha=None, epsilon=0):
    n = np.shape(Y)[0]
    sum_i = 0
    if MinvXbeta is None:
        if MinvX is None:
            MinvX = np.linalg.solve(M,X)
        MinvXbeta = np.matmul(MinvX, beta)
    WMinvXbeta = np.matmul(W,MinvXbeta)
    U = np.linalg.solve(M,WMinvXbeta)
    if alpha is None:
        alpha = mu*phi[:,None]
    for i in range(n):
        sum_omega = np.sum(mu[i]*U[i])
        digamma_alpha = digamma(alpha[i])
        sum_i += phi[i]*np.sum( mu[i]*( np.log(Y[i]+epsilon)*(U[i]-sum_omega) - U[i]*(digamma_alpha-np.sum(mu[i]*digamma_alpha)) ) )
    return(sum_i)


def dirichlet_second_derivative_wrt_rho(mu, phi, Minv, beta, W, X, Y, Z, MinvX=None, alpha=None, epsilon=0):
    n = np.shape(Y)[0]
    sum_i = 0
    if MinvX is None:
        MinvXbeta = np.matmul(Minv,np.matmul(X,beta))
    else:
        MinvXbeta = np.matmul(MinvX,beta)
    MinvW = np.matmul(Minv,W)
    U = np.matmul(MinvW, MinvXbeta)
    V = np.matmul(MinvW, U)
    if alpha is None:
        alpha = np.copy(mu)
        for j in range(J):
            alpha[:,j] = phi*mu[:,j]
    for i in range(n):
        omega = mu[i]*U[i]
        sum_omega = np.sum(omega)
        digamma_alpha = digamma(alpha[i])
        trigamma_alpha = polygamma(1,alpha[i])
        logYi = np.log(Y[i]+epsilon)
        
        Fi = logYi*(U[i]-sum_omega)
        Gi = U[i]*(digamma_alpha-np.sum(mu[i]*digamma_alpha))
        term_1 = np.sum( (omega - mu[i]*sum_omega) * ( Fi - Gi ) )
        
        dFi = logYi * ( 2*V[i] - np.sum(mu[i]*(2*V[i]+U[i]**2)) + sum_omega**2 )
        dGi = 2*V[i]*(digamma_alpha - np.sum(mu[i]*digamma_alpha)) + U[i]*( (omega-mu[i]*sum_omega)*(phi[i]*trigamma_alpha) - np.sum( (omega-mu[i]*sum_omega)*(digamma_alpha + phi[i]*mu[i]*trigamma_alpha) ) )
        term_2 = np.sum( mu[i] * (dFi - dGi) )
        
        sum_i += phi[i] * (term_1 + term_2)
        #sum_i += phi[i] * np.sum( (omega - mu[i]*sum_omega) * ( Fi - Gi ) +  mu[i] * (dFi - dGi) )
    return sum_i


def dirichlet_second_derivative_wrt_rho_gamma(mu, phi, Minv, beta, W, X, Y, Z, MinvX=None, alpha=None, epsilon=0):
    n = np.shape(Y)[0]
    K = np.shape(X)[1]
    if MinvX is None:
        MinvXbeta = np.matmul(Minv,np.matmul(X,beta))
    else:
        MinvXbeta = np.matmul(MinvX,beta)
    U = np.matmul(np.matmul(Minv,W), MinvXbeta)
    derivative_rho_gamma = np.zeros(K)
    if alpha is None:
        alpha = np.copy(mu)
        for j in range(J):
            alpha[:,j] = phi*mu[:,j]
    for i in range(n):
        sum_omega = np.sum(mu[i]*U[i])
        digamma_alpha = digamma(alpha[i])
        trigamma_alpha = polygamma(1,alpha[i])
        derivative_rho_gamma += Z[i,:]*phi[i]*np.sum( mu[i]*( np.log(Y[i]+epsilon)*(U[i]-sum_omega) - U[i]*(digamma_alpha-np.sum(mu[i]*digamma_alpha))  
                                                             - phi[i]*U[i]*(mu[i]*trigamma_alpha - np.sum(mu[i]**2*trigamma_alpha)) ) )
    return(derivative_rho_gamma)


def dirichlet_second_derivative_wrt_rho_beta(mu, phi, Minv, beta, W, X, Y, Z, MinvX=None, alpha=None, epsilon=0):
    n = np.shape(Y)[0]
    K = np.shape(X)[1]
    J = np.shape(Y)[1]
    if MinvX is None:
        MinvX = np.matmul(Minv,X)
    MinvXbeta = np.matmul(MinvX,beta)
    MinvW = np.matmul(Minv,W)
    U = np.matmul(MinvW, MinvXbeta)
    Q = np.matmul(MinvW,MinvX)
    hessian = np.zeros((K,J))
    if alpha is None:
        alpha = np.copy(mu)
        for j in range(J):
            alpha[:,j] = phi*mu[:,j]
    for i in range(n):
        omega = mu[i]*U[i]
        sum_omega = np.sum(omega)
        digamma_alpha = digamma(alpha[i])
        trigamma_alpha = polygamma(1,alpha[i])
        logYi = np.log(Y[i]+epsilon)
        Fi = logYi*(U[i]-sum_omega)
        Gi = U[i]*(digamma_alpha-np.sum(mu[i]*digamma_alpha))
        for p in range(K):
            for d in range(J):
                term_1 = MinvX[i,p]*( (Fi[d]-Gi[d]) - np.sum(mu[i]*(Fi-Gi)) )
                term_2 = logYi[d]*Q[i,p] - ( Q[i,p] + MinvX[i,p]*(U[i,d]-sum_omega) )*np.sum(mu[i]*logYi)
                term_3 = - Q[i,p]*(digamma_alpha[d] - np.sum(mu[i]*digamma_alpha)) - MinvX[i,p]*phi[i]*(omega[d]*trigamma_alpha[d] - np.sum(mu[i]**2*U[i]*trigamma_alpha ) )
                term_4 = MinvX[i,p]*sum_omega*( digamma_alpha[d] + alpha[i,d]*trigamma_alpha[d] - np.sum(mu[i]*(digamma_alpha + alpha[i]*trigamma_alpha)) )

                hessian[p,d] += phi[i] * mu[i,d] * (term_1 + term_2 + term_3 + term_4)
    return hessian

##### NO SPATIAL #####

def gradientDescent(X, Y, beta_0, learning_rate=0.01, nitermax=1000):
    current_beta = np.copy(beta_0)
    list_likelihoods = []
    list_gradients = []
    list_betas = [current_beta]

    for i in range(nitermax):
        current_mu = compute_mu(X, current_beta)
        list_likelihoods.append(dirichlet_loglikelihood(current_mu,Y))
        
        grad = dirichlet_gradient(current_mu, X, Y)
        list_gradients.append(grad)
        # we want to maximize the function, so we add the gradient
        current_beta = current_beta + learning_rate * grad
        current_beta[:,0] = 0
        list_betas.append(current_beta)
        
    return(list_betas, list_likelihoods, list_gradients)


##### SPATIAL #####

def gradientDescentSpatial(X, Y, W, beta_0, rho_0, learning_rate=0.01, nitermax=1000):
    current_beta = np.copy(beta_0)
    current_rho = rho_0
    list_likelihoods = []
    list_gradients = []
    list_betas = [current_beta]
    list_rhos = [current_rho]
    
    n = np.shape(X)[0]
    I_n = np.identity(n)

    for i in range(nitermax):
        Minv = np.linalg.inv(I_n - current_rho*W)
        MinvX = np.matmul(Minv,X)
        current_mu = compute_mu_spatial(X, current_beta, MinvX=MinvX)
        list_likelihoods.append(dirichlet_loglikelihood(current_mu,Y))
        
        grad = dirichlet_gradient(current_mu, MinvX, Y)
        derivative_rho = dirichlet_derivative_wrt_rho(current_mu, Minv, current_beta, W, X, Y, MinvX=MinvX)
        list_gradients.append([grad,derivative_rho])
        # we want to maximize the function, so we add the gradient
        current_beta = current_beta + learning_rate * grad
        current_beta[:,0] = 0
        list_betas.append(current_beta)
        current_rho = current_rho + learning_rate*derivative_rho
        list_rhos.append(current_rho)
        
    return(list_betas, list_rhos, list_likelihoods, list_gradients)


####################

class dirichletRegressor:
    
    def __init__(self, learning_rate=0.01, nitermax=1000, random_state=None, spatial=False):
        self.learning_rate = learning_rate
        self.nitermax = nitermax
        self.random_state = random_state
        self.spatial = spatial
        
    def fit(self, X , Y, W=None, beta_0=None, rho_0=0):
        self.X = X
        self.Y = Y
        
        if beta_0 is None:
            beta_0 = np.zeros((np.shape(X)[1], np.shape(Y)[1]))
            
        self.beta_0 = beta_0
        self.rho_0 = rho_0
        
        if self.spatial:
            if W is None: # we create W by taking the 5 nearest neighbors in X
                neighbors = NearestNeighbors(n_neighbors=6).fit(X)
                W = neighbors.kneighbors_graph(X).toarray()
                # replace the 1 on the diagonal by 0
                np.fill_diagonal(W,0)
                # scaling the matrix, so that the sum of each row is 1
                W = W/5
            list_betas, list_rhos, list_likelihoods, list_gradients = gradientDescentSpatial(self.X, self.Y, W, 
                                                                                             beta_0, rho_0, 
                                                                                             learning_rate=self.learning_rate, 
                                                                                             nitermax=self.nitermax)
            self.rho = list_rhos[-1]
            self.list_rhos = list_rhos
            
        else: 
            list_betas, list_likelihoods, list_gradients = gradientDescent(self.X, self.Y, beta_0, 
                                                                           learning_rate=self.learning_rate, 
                                                                           nitermax=self.nitermax)
        
        self.beta = list_betas[-1]
        self.list_betas = list_betas
        self.list_likelihoods = list_likelihoods
        self.list_gradients = list_gradients
        
    

    
#################################### OLD STUFF ####################################

def dirichlet_derivative(p,d,mu,X,Y):
    n = np.shape(Y)[0]
    sum_i = 0
    for i in range(n):
        sum_mu_ij = np.sum(mu[i])
        digamma_sum_mu_ij = digamma(sum_mu_ij) 
        digamma_mu = digamma(mu[i])
        sum_i += X[i,p]*mu[i,d]*( digamma_sum_mu_ij 
                        + np.sum(mu[i] * digamma_mu) - sum_mu_ij * digamma_sum_mu_ij - np.sum(mu[i]*np.log(Y[i]+1e-200)) 
                       - digamma_mu[d] + np.log(Y[i,d]+1e-200) )
    return(sum_i)


def dirichlet_gradient_old(mu, X, Y):
    K = np.shape(X)[1] #nb of features
    J = np.shape(Y)[1] #nb of classes
    n = np.shape(Y)[0] 
    gradient = np.zeros((K,J))
    for p in range(K):
        for d in range(J):
            gradient[p,d] = dirichlet_derivative(p,d,mu,X,Y)
    return gradient


def dirichlet_gradient(mu, X, Y, epsilon=0):
    K = np.shape(X)[1] #nb of features
    J = np.shape(Y)[1] #nb of classes
    n = np.shape(Y)[0] 
    gradient = np.zeros((K,J))
    for i in range(n):
        sum_mu_ij = np.sum(mu[i])
        digamma_sum_mu_ij = digamma(sum_mu_ij) 
        digamma_mu = digamma(mu[i])
        sum_mu_digamma_mu = np.sum(mu[i] * digamma_mu)
        logY = np.log(Y[i] + epsilon)
        sum_mu_logY = np.sum(mu[i]*logY)
        for p in range(K):
            Xip = X[i,p]
            gradient[p,:] += Xip*mu[i,:]*( digamma_sum_mu_ij + sum_mu_digamma_mu
                            - sum_mu_ij * digamma_sum_mu_ij - sum_mu_logY
                           - digamma_mu + logY )
    return gradient

def dirichlet_derivative_wrt_rho_old(mu, phi, beta, W, X, Y, Z, MinvX=None, alpha=None, epsilon=0):
    n = np.shape(Y)[0]
    sum_i = 0
    if MinvX is None:
        MinvXbeta = np.matmul(Minv,np.matmul(X,beta))
    else:
        MinvXbeta = np.matmul(MinvX,beta)
    U = np.matmul(np.matmul(Minv,W), MinvXbeta)
    if alpha is None:
        alpha = mu*phi[:,None]
    for i in range(n):
        sum_omega = np.sum(mu[i]*U[i])
        digamma_alpha = digamma(alpha[i])
        sum_i += phi[i]*np.sum( mu[i]*( np.log(Y[i]+epsilon)*(U[i]-sum_omega) - U[i]*(digamma_alpha-np.sum(mu[i]*digamma_alpha)) ) )
    return(sum_i)