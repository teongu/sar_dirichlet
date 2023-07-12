import numpy as np
import pandas as pd
from smote_cd.dataset_generation import softmax
from scipy.special import gamma, digamma, polygamma, loggamma
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from scipy.optimize import minimize, Bounds



def compute_mu(X, beta, Xbeta=None):
    """
    Compute the log-odds probabilities vector mu.

    Parameters:
        X (ndarray): Input matrix of shape (n, K) with n the number of samples and K the number of features.
        beta (ndarray): Coefficient matrix of shape (K, J) where J is the number of classes.
        Xbeta (ndarray, optional): Precomputed matrix product X * beta. If not provided, it will be computed.

    Returns:
        ndarray: Probabilities matrix mu of shape (n, J) where n is the number of samples and J is the number of classes.
    """
    if Xbeta is None:
        Xbeta = np.matmul(X,beta)
    exp_Xbeta = np.exp(Xbeta - np.max(Xbeta, axis=1, keepdims=True))
    sum_exp_Xbeta = np.sum(exp_Xbeta, axis=1, keepdims=True)
    mu = exp_Xbeta / sum_exp_Xbeta
    mu[mu==0] = 1e-305  # Replace 0 with a small positive value
    return mu


def compute_mu_spatial(X, beta, M, Xbeta=None, MinvX=None, MXbeta=None):
    """
    Compute the log-odds probabilities vector mu with spatial correlation.

    Parameters:
        X (ndarray): Input matrix of shape (n, K) with n the number of samples and K the number of features.
        beta (ndarray): Coefficient matrix of shape (K, J) where J is the number of classes.
        M (ndarray or sparse matrix): Spatial correlation matrix of shape (n, n) or sparse matrix format.
        Xbeta (ndarray, optional): Precomputed matrix product X * beta. If not provided, it will be computed.
        MinvX (ndarray, optional): Precomputed matrix product M_inverse * X. If not provided, it will be computed.
        MXbeta (ndarray, optional): Precomputed matrix product M_inverse * X * beta. If not provided, it will be computed.

    Returns:
        ndarray: Probabilities matrix mu of shape (n, J) where n is the number of samples and J is the number of classes.
    """
    if MXbeta is None:
        if MinvX is None:
            if Xbeta is None:
                Xbeta = np.matmul(X,beta)
            MXbeta = sparse.linalg.spsolve(sparse.csc_matrix(M),Xbeta)
        else:
            MXbeta = np.matmul(MinvX,beta)
    exp_MXbeta = np.exp(MXbeta - np.max(MXbeta, axis=1, keepdims=True))
    sum_exp_MXbeta = np.sum(exp_MXbeta, axis=1, keepdims=True)
    mu = exp_MXbeta / sum_exp_MXbeta
    mu[mu==0]=1e-305
    return mu

####################


def dirichlet_loglikelihood(mu,phi,Y,epsilon=0):
    """
    Compute the log-likelihood of Dirichlet distribution given parameters.

    Parameters:
        mu (ndarray): Probability matrix of shape (n, J) representing Dirichlet distribution parameters,
                      where n is the number of samples and J is the number of classes.
        phi (ndarray): Concentration parameter array of shape (J,) representing Dirichlet distribution parameters.
        Y (ndarray): Observation matrix of shape (n, J) representing the observed values,
                     where n is the number of samples and J is the number of classes.
        epsilon (float, optional): Small value added to Y to avoid log(0) calculations. Default is 0.

    Returns:
        float: Log-likelihood of the Dirichlet distribution.

    Notes:
        The log-likelihood is computed based on the formula:
            sum(loggamma(phi)) - sum(loggamma(phi_mu)) + sum((phi_mu - 1) * log(Y + epsilon))
        where phi_mu = phi.reshape(-1, 1) * mu.
    """
    phi_mu = phi.reshape(-1,1) * mu
    sum_loggamma_phi_mu = np.sum(loggamma(phi_mu),axis=1)
    sum_phi_mu_times_logY = np.sum( (phi_mu-1)*np.log(Y+epsilon) , axis=1 )
    sum_ll = np.sum(loggamma(phi) - sum_loggamma_phi_mu + sum_phi_mu_times_logY)
    return sum_ll


def gradient_wrt_beta(mu, phi, X, Y, epsilon=0):
    K = np.shape(X)[1] #nb of features
    J = np.shape(Y)[1] #nb of classes
    n = np.shape(Y)[0] 
    gradient = np.zeros((K,J))
    sum_mu = np.sum(mu, axis=1)
    digamma_sum_mu = digamma(sum_mu)
    digamma_phi_mu = digamma(phi.reshape(-1,1)*mu)
    logY = np.log(Y + epsilon)
    sum_mu_digamma_phi_mu_minus_sum_mu_logY = np.sum(mu * digamma_phi_mu, axis=1) - np.sum(mu*logY,axis=1)
    logY_minus_digamma_phi_mu = np.log(Y + epsilon) - digamma_phi_mu
    mu_phi_times_sum_terms = mu * phi.reshape(-1,1)*(sum_mu_digamma_phi_mu_minus_sum_mu_logY.reshape(-1,1) + logY_minus_digamma_phi_mu)
    gradient = np.einsum('ni,nj->ij', X, mu_phi_times_sum_terms)
    return gradient


def gradient_wrt_gamma(mu, phi, Y, Z, epsilon=0):
    mu_logY_minus_digamma_phi_mu = mu * ( np.log(Y+epsilon) - digamma(phi.reshape(-1,1)*mu) )
    phi_times_digamma_phi_plus_sum = phi * ( digamma(phi) + np.sum(mu_logY_minus_digamma_phi_mu, axis=1) )
    derivative_gamma = np.sum(Z * phi_times_digamma_phi_plus_sum.reshape(-1,1), axis=0)
    return(derivative_gamma)


def derivative_wrt_rho(mu, phi, beta, M, W, X, Y, MinvX=None, MinvXbeta=None, alpha=None, epsilon=0):
    if MinvXbeta is None:
        if MinvX is None:
            MinvX = np.linalg.solve(M,X)
        MinvXbeta = np.matmul(MinvX, beta)
    WMinvXbeta = np.matmul(W,MinvXbeta)
    U = np.linalg.solve(M,WMinvXbeta)
    if alpha is None:
        alpha = mu*phi[:,None]
    sum_omega = np.sum(mu*U,axis=1)
    digamma_alpha = digamma(alpha)
    sum_mu_digamma_alpha = np.sum(mu*digamma_alpha, axis=1)
    U_digamma_alpha_times_sum_mu_digamma_alpha = U*(digamma_alpha-sum_mu_digamma_alpha.reshape(-1,1))
    logY_U_sum_omega = np.log(Y+epsilon)*(U-sum_omega.reshape(-1,1))
    mu_times_subtract = mu*(logY_U_sum_omega - U_digamma_alpha_times_sum_mu_digamma_alpha)
    return( np.sum(phi * np.sum(mu_times_subtract,axis=1)) )


def hessian_wrt_gamma(mu, phi, beta, X, Y, Z, epsilon=0):
    K = np.shape(Z)[1]
    n = np.shape(Y)[0]
    hessian = np.zeros((K,K))
    phi_mu = phi.reshape(-1,1) * mu
    logY = np.log(Y+epsilon)
    phi_trigamma_plus_digamma = phi*polygamma(1,phi) + digamma(phi)
    phi_sum_mu_2_trigamma = phi * np.sum( mu**2 * polygamma(1,phi_mu), axis=1) 
    sum_mu_times_logY_minus_digamma = np.sum(mu*(logY-digamma(phi_mu)), axis=1) 
    Z_phi_times_terms = Z * (phi * (phi_trigamma_plus_digamma - phi_sum_mu_2_trigamma + sum_mu_times_logY_minus_digamma)).reshape(-1,1)
    for k in range(K):
        hessian[:,k] = np.sum(Z[:,k].reshape(-1,1) * Z_phi_times_terms, axis=0)
    return(hessian)


def hessian_wrt_beta(mu, phi, X, Y, alpha=None, epsilon=0):
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
        phi_sum_mu_2_trigamma = phi[i]*np.sum(mu_2_trigamma)
        logY = np.log(Y[i]+epsilon)
        sum_mu_log = 2*np.sum(mu[i]*logY)
        mu_digamma_alpha = mu[i]*digamma_alpha
        sum_mu_digamma = 2*np.sum(mu_digamma_alpha)    
        mu_digamma_log = mu[i]*(digamma_alpha-logY)
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

    return hessian



def second_derivative_beta_gamma(mu, phi, beta, X, Y, Z, epsilon=0):
    K = np.shape(X)[1]
    J = np.shape(Y)[1]
    K_Z = Z.shape[-1]
    n = np.shape(Y)[0]
    derivatives = np.zeros((K_Z,K,J))
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


def second_derivative_wrt_rho(mu, phi, beta, M, W, X, Y, MinvX=None, MinvXbeta=None, alpha=None, epsilon=0):
    n = np.shape(Y)[0]
    if MinvXbeta is None:
        if MinvX is None:
            MinvX = np.linalg.solve(M,X)
        MinvXbeta = np.matmul(MinvX, beta)
    WMinvXbeta = np.matmul(W,MinvXbeta)
    U = np.linalg.solve(M,WMinvXbeta)
    V = np.linalg.solve(M,np.matmul(W, U))
    if alpha is None:
        alpha = mu*phi[:,None]
    sum_i = 0
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


def second_derivative_wrt_rho_gamma(mu, phi, beta, M, W, X, Y, Z, MinvX=None, MinvXbeta=None, alpha=None, epsilon=0):
    n = np.shape(Y)[0]
    K_Z = np.shape(Z)[1]
    if MinvXbeta is None:
        if MinvX is None:
            MinvX = np.linalg.solve(M,X)
        MinvXbeta = np.matmul(MinvX, beta)
    WMinvXbeta = np.matmul(W,MinvXbeta)
    U = np.linalg.solve(M,WMinvXbeta)
    derivative_rho_gamma = np.zeros(K_Z)
    if alpha is None:
        alpha = mu*phi[:,None]
    for i in range(n):
        sum_omega = np.sum(mu[i]*U[i])
        digamma_alpha = digamma(alpha[i])
        trigamma_alpha = polygamma(1,alpha[i])
        derivative_rho_gamma += Z[i,:]*phi[i]*np.sum( mu[i]*( np.log(Y[i]+epsilon)*(U[i]-sum_omega) - U[i]*(digamma_alpha-np.sum(mu[i]*digamma_alpha)) - phi[i]*U[i]*(mu[i]*trigamma_alpha - np.sum(mu[i]**2*trigamma_alpha)) ) )
    return(derivative_rho_gamma)


def second_derivative_wrt_rho_beta(mu, phi, beta, M, W, X, Y, MinvX=None, MinvXbeta=None, alpha=None, epsilon=0):
    n = np.shape(Y)[0]
    K = np.shape(X)[1]
    J = np.shape(Y)[1]
    if MinvX is None:
        MinvX = np.linalg.solve(M,X)
    WMinvX = np.matmul(W, MinvX)
    WMinvXbeta = np.matmul(WMinvX,beta)
    Q = np.linalg.solve(M,WMinvX)
    U = np.matmul(Q,beta)
    hessian = np.zeros((K,J))
    if alpha is None:
        alpha = mu*phi[:,None]
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


def compute_gradient_spatial(mu, phi, beta, M, W, X, Y, Z=None):
    K = X.shape[-1]
    J = Y.shape[-1]
    n = len(X)
    
    MinvX = np.linalg.solve(M,X)

    if Z is None:
        grad = gradient_wrt_beta(mu, phi, MinvX, Y)[:,1:].flatten()
    else:
        K_gamma = Z.shape[-1]
        grad = np.zeros(K*(J-1)+K_gamma)
        grad[:K*(J-1)] = gradient_wrt_beta(mu, phi, MinvX, Y)[:,1:].flatten()
        grad[K*(J-1):] = gradient_wrt_gamma(mu, phi, Y, Z)

    grad_spatial = np.zeros(len(grad)+1)
    grad_spatial[:-1] = grad
    grad_spatial[-1] = derivative_wrt_rho(mu, phi, beta, M, W, X, Y, MinvX)
    return(grad_spatial)
    
def compute_hessian_spatial(mu, phi, beta, M, W, X, Y, Z=None):
    K = X.shape[-1]
    J = Y.shape[-1]
    n = len(X)
    
    MinvX = np.linalg.solve(M,X)

    if Z is None:
        hess = hessian_wrt_beta(mu, phi, MinvX, Y)[:,1:,:,1:].reshape((K*(J-1),K*(J-1)))
    else:
        K_gamma = Z.shape[-1]
        hess = np.zeros((K*(J-1) + K_gamma, K*(J-1) + K_gamma))
        hess[:-K_gamma,:-K_gamma] = hessian_wrt_beta(mu, phi, MinvX, Y)[:,1:,:,1:].reshape((K*(J-1),K*(J-1)))
        hess[-K_gamma:,-K_gamma:] = hessian_wrt_gamma(mu, phi, beta, MinvX, Y, Z)
        der_beta_gamma = second_derivative_beta_gamma(mu, phi, beta, MinvX, Y, Z)[:,:,1:].reshape((K_gamma,K*(J-1)))
        hess[-K_gamma:,:-K_gamma] = der_beta_gamma
        hess[:-K_gamma,-K_gamma:] = np.transpose(der_beta_gamma)

    hess_spatial = np.zeros((hess.shape[0]+1,hess.shape[1]+1))
    hess_spatial[:-1,:-1] = hess
    hess_spatial[-1,-1] = second_derivative_wrt_rho(mu, phi, beta, M, W, X, Y)
    der_rho_beta = second_derivative_wrt_rho_beta(mu, phi, beta, M, W, X, Y)[:,1:].flatten()
    hess_spatial[-1,:K*(J-1)] = der_rho_beta
    hess_spatial[:K*(J-1),-1] = der_rho_beta
    if not Z is None:
        der_rho_gamma = second_derivative_wrt_rho_gamma(mu, phi, beta, M, W, X, Y, Z)
        hess_spatial[-1,K*(J-1):-1] = der_rho_gamma
        hess_spatial[K*(J-1):-1,-1] = der_rho_gamma
        
    return(hess_spatial)
            
####################

def crossentropy_no_spatial(x, X, Y, regularization_lambda=0, epsilon=0, size_samples=None):
    K = X.shape[-1]
    J = Y.shape[-1]
    n = X.shape[0]
    beta = np.zeros((K,J))
    beta[:,1:] = x.reshape((K,J-1))
    mu = compute_mu(X, beta)
    if size_samples is None:
        obj = - 1/n * np.sum(Y*np.log(mu+epsilon))
    else:
        obj = -1/np.sum(size_samples) * np.sum( size_samples*np.sum(Y*np.log(mu+epsilon),axis=1) )
    if regularization_lambda!=0:
        obj = obj + regularization_lambda * np.linalg.norm(x)**2
    return obj

def jac_crossentropy_no_spatial(x, X, Y, regularization_lambda=0, epsilon=0):
    K = X.shape[-1]
    J = Y.shape[-1]
    n = X.shape[0]
    beta = np.zeros((K,J))
    beta[:,1:] = x.reshape((K,J-1))
    mu = compute_mu(X, beta)
    
    term = Y - mu*np.sum(Y, axis=1).reshape(-1,1)
    grad = np.einsum('ni,nj->ij', X, term)
        
    return -1/n * grad[:,1:].flatten() + 2*regularization_lambda*x


def crossentropy_spatial(x, X, Y, W, regularization_lambda=0, epsilon=0):
    K = X.shape[-1]
    J = Y.shape[-1]
    n = X.shape[0]
    beta = np.zeros((K,J))
    beta[:,1:] = x[:-1].reshape((K,J-1))
    rho = x[-1]
    M = np.identity(n) - rho*W
    mu = compute_mu_spatial(X, beta, M)
    obj = - 1/n * np.sum(Y*np.log(mu+epsilon))
    if regularization_lambda!=0:
        obj = obj + regularization_lambda * np.linalg.norm(x)**2
    return obj

def jac_crossentropy_spatial(x, X, Y, W, regularization_lambda=0, epsilon=0):
    K = X.shape[-1]
    J = Y.shape[-1]
    n = X.shape[0]
    beta = np.zeros((K,J))
    beta[:,1:] = x[:-1].reshape((K,J-1))
    rho = x[-1]
    M = np.identity(n) - rho*W
    MinvX = np.linalg.solve(M,X)
    mu = compute_mu_spatial(X, beta, M, MinvX=MinvX)
    
    term = Y - mu*np.sum(Y, axis=1).reshape(-1,1)
    grad = np.einsum('ni,nj->ij', MinvX, term)
    
    MinvXbeta = np.matmul(MinvX, beta)
    WMinvXbeta = np.matmul(W,MinvXbeta)
    U = np.linalg.solve(M,WMinvXbeta)
    sum_omega = np.sum(mu*U,axis=1)
    derivative_rho = np.sum(Y * (U - sum_omega.reshape(-1,1)))
    
    return -1/n * np.concatenate([grad[:,1:].flatten(),[derivative_rho]]) + 2*regularization_lambda*x

def objective_func_loglik_no_spatial(x, X, Y, Z=None, phi=None, regularization_lambda=0, epsilon=0):
    K = X.shape[-1]
    J = Y.shape[-1]
    n = X.shape[0]
    beta = np.zeros((K,J))
    if phi is None:
        beta[:,1:K*(J-1)] = x[:K*(J-1)].reshape((K,J-1))
        gamma_var = x[K*(J-1):]
        phi = np.exp(np.matmul(Z,gamma_var))
    else:
        beta[:,1:] = x.reshape((K,J-1))
    mu = compute_mu(X, beta)
    obj = - 1/n * dirichlet_loglikelihood(mu,phi,Y,epsilon=epsilon)
    if regularization_lambda!=0:
        obj = obj + regularization_lambda * np.linalg.norm(x)**2
    return obj

def jac_objective_func_loglik_no_spatial(x, X, Y, Z=None, phi=None, regularization_lambda=0, epsilon=0):
    K = X.shape[-1]
    J = Y.shape[-1]
    n = X.shape[0]
    beta = np.zeros((K,J))
    if phi is None:
        beta[:,1:K*(J-1)] = x[:K*(J-1)].reshape((K,J-1))
        gamma_var = x[K*(J-1):]
        phi = np.exp(np.matmul(Z,gamma_var))
    else:
        beta[:,1:] = x.reshape((K,J-1))
    mu = compute_mu(X, beta)
    beta_grad = gradient_wrt_beta(mu, phi, X, Y, epsilon=epsilon)[:,1:]

    if Z is None:
        return -1/n * beta_grad.flatten() + 2*regularization_lambda*x
    else:
        gamma_grad = gradient_wrt_gamma(mu, phi, Y, Z, epsilon=epsilon)
        return(-1/n * np.concatenate([beta_grad.flatten(),gamma_grad]) + 2*regularization_lambda*x)

    
def objective_func_loglik_spatial(x, X, Y, W, Z=None, phi=None, regularization_lambda=0, epsilon=0):
    K = X.shape[-1]
    J = Y.shape[-1]
    n = X.shape[0]
    beta = np.zeros((K,J))
    if phi is None:
        beta[:,1:K*(J-1)] = x[:K*(J-1)].reshape((K,J-1))
        gamma_var = x[K*(J-1):-1]
        phi = np.exp(np.matmul(Z,gamma_var))
    else:
        beta[:,1:] = x[:-1].reshape((K,J-1))
    rho = x[-1]
    M = np.identity(n) - rho*W
    mu = compute_mu_spatial(X, beta, M)
    obj = - 1/n * dirichlet_loglikelihood(mu,phi,Y,epsilon=epsilon)
    if regularization_lambda!=0:
        obj = obj + regularization_lambda * np.linalg.norm(x)**2
    return obj

def jac_objective_func_loglik_spatial(x, X, Y, W, Z=None, phi=None, regularization_lambda=0, epsilon=0):
    K = X.shape[-1]
    J = Y.shape[-1]
    n = X.shape[0]
    beta = np.zeros((K,J))
    if phi is None:
        beta[:,1:K*(J-1)] = x[:K*(J-1)].reshape((K,J-1))
        gamma_var = x[K*(J-1):-1]
        phi = np.exp(np.matmul(Z,gamma_var))
    else:
        beta[:,1:] = x[:-1].reshape((K,J-1))
    rho = x[-1]
    M = np.identity(n) - rho*W
    MinvX = np.linalg.solve(M,X)
    mu = compute_mu_spatial(X, beta, M, MinvX=MinvX)
    
    beta_grad = gradient_wrt_beta(mu, phi, MinvX, Y, epsilon=epsilon)[:,1:]
    
    MinvW = np.linalg.solve(M,W)
    rho_derivative = derivative_wrt_rho(mu, phi, beta, M, W, X, Y, MinvX=None, epsilon=epsilon)

    if Z is None:
        return (-1/n * np.concatenate([beta_grad.flatten(),[rho_derivative]]) + 2*regularization_lambda*x)
    else:
        gamma_grad = gradient_wrt_gamma(mu, phi, Y, Z, epsilon=epsilon)
        return(-1/n * np.concatenate([beta_grad.flatten(),gamma_grad,[rho_derivative]]) + 2*regularization_lambda*x)

    
    
####################

class dirichletRegressor:
    
    def __init__(self, learning_rate=0.01, maxiter=1000, random_state=None, spatial=False, maxfun=1000):
        self.learning_rate = learning_rate
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.random_state = random_state
        self.spatial = spatial
        
    def fit(self, X , Y, W=None, beta_0=None, rho_0=0, parametrization='common', gamma_0=None, Z=None, fit_intercept=True, regularization=0, verbose=True, loss='likelihood', size_samples=None):
        
        n,K = np.shape(X)
        J = Y.shape[1]
        if fit_intercept:
            K+=1
            self.intercept = True
        else:
            self.intercept=False
        self.K, self.J = K, J
        
        if beta_0 is None:
            beta_0 = np.zeros((K,J-1))
        if parametrization=='common':
            self.phi = np.ones(n)
        else:
            if gamma_0 is None:
                gamma_0 = np.zeros(Z.shape[1])
                
        if fit_intercept:
            X_f = np.ones((n,K))
            X_f[:,1:] = X
        else:
            X_f = np.copy(X)
                
        if self.spatial:
            if loss=='crossentropy':
                params0 = np.concatenate([beta_0.flatten(),[rho_0]])
                min_bounds, max_bounds = -np.inf*np.ones(len(params0)), np.inf*np.ones(len(params0))
                min_bounds[-1], max_bounds[-1] = -1, 1
                bounds = Bounds(min_bounds, max_bounds)
                solution = minimize(crossentropy_spatial, params0, args=(X_f, Y, W, regularization), bounds=bounds, jac = jac_crossentropy_spatial)
                self.beta = solution.x[:-1].reshape((K,J-1))
            else:
                if parametrization=='common':
                    params0 = np.concatenate([beta_0.flatten(),[rho_0]])
                    min_bounds, max_bounds = -np.inf*np.ones(len(params0)), np.inf*np.ones(len(params0))
                    min_bounds[-1], max_bounds[-1] = -1, 1
                    bounds = Bounds(min_bounds, max_bounds)
                    solution = minimize(objective_func_loglik_spatial, params0, args=(X_f, Y, W, None, self.phi, regularization), bounds=bounds, options={'maxiter':self.maxiter, 'maxfun':self.maxfun})
                    self.beta = solution.x[:-1].reshape((K,J-1))

                else:
                    params0 = np.concatenate([beta_0.flatten(),gamma_0,[rho_0]])
                    min_bounds, max_bounds = -np.inf*np.ones(len(params0)), np.inf*np.ones(len(params0))
                    min_bounds[-1], max_bounds[-1] = -1, 1
                    bounds = Bounds(min_bounds, max_bounds)
                    solution = minimize(objective_func_loglik_spatial, params0, args=(X_f, Y, W, Z, None, regularization), bounds=bounds, options={'maxiter':self.maxiter, 'maxfun':self.maxfun})
                    self.beta = solution.x[:K*(J-1)].reshape((K,J-1))
                    self.gamma = solution.x[K*(J-1):-1]
                    self.phi = np.exp(np.matmul(Z,self.gamma))
            self.rho = solution.x[-1]
            beta = np.zeros((K, J))
            beta[:,1:] = self.beta
            M = np.identity(n) - self.rho*W
            self.mu = compute_mu_spatial(X_f, beta, M)
            
        else: 
            if loss=='crossentropy':
                params0 = beta_0.flatten()
                if size_samples is None:
                    solution = minimize(crossentropy_no_spatial, params0, args=(X_f, Y, regularization), jac = jac_crossentropy_no_spatial)
                else:
                    solution = minimize(crossentropy_no_spatial, params0, args=(X_f, Y, regularization, 0, size_samples))
                self.beta = solution.x.reshape((K,J-1))
            else:
                if parametrization=='common':
                    params0 = beta_0.flatten()
                    #solution = minimize(objective_func_loglik_no_spatial, params0, args=(X_f,Y,None,self.phi,regularization))
                    solution = minimize(objective_func_loglik_no_spatial, params0, args=(X_f,Y,None,self.phi,regularization), jac = jac_objective_func_loglik_no_spatial)
                    self.beta = solution.x.reshape((K,J-1))
                else:
                    params0 = np.concatenate([beta_0.flatten(),gamma_0])
                    solution = minimize(objective_func_loglik_no_spatial, params0, args=(X_f,Y,Z,None,regularization), jac = jac_objective_func_loglik_no_spatial)
                    self.beta = solution.x[:K*(J-1)].reshape((K,J-1))
                    self.gamma = solution.x[K*(J-1):]
                    self.phi = np.exp(np.matmul(Z,self.gamma))
                
            beta = np.zeros((K, J))
            beta[:,1:] = self.beta
            self.mu = compute_mu(X_f, beta)
                
        if verbose:
            print(solution.message)
            
            
    def compute_hessian(self, X, Y, Z=None, W=None):
        beta = np.zeros((self.K, self.J))
        beta[:,1:] = self.beta
        n = len(X)
        if self.intercept:
            X_f = np.ones((len(X),self.K))
            X_f[:,1:] = X
        else:
            X_f = X
        if self.spatial:
            M = np.identity(n) - self.rho*W
            mu = compute_mu_spatial(X_f, beta, M)
        else:
            mu = compute_mu(X_f, beta)

        if hasattr(self,'gamma'):
            phi = np.exp(np.matmul(Z,self.gamma))
            K_gamma = len(self.gamma)
            hess = np.zeros((self.K*self.J + K_gamma, self.K*self.J + K_gamma))
            hess[:-K_gamma,:-K_gamma] = hessian_wrt_beta(mu, phi, X_f, Y).reshape((self.K*self.J,self.K*self.J))
            hess[-K_gamma:,-K_gamma:] = hessian_wrt_gamma(mu, phi, beta, X_f, Y, Z)
            der_beta_gamma = second_derivative_beta_gamma(mu, phi, beta, X_f, Y, Z).reshape((K_gamma,self.K*self.J))
            hess[-K_gamma:,:-K_gamma] = der_beta_gamma
            hess[:-K_gamma,-K_gamma:] = np.transpose(der_beta_gamma)
        else:
            phi = np.ones(n)
            hess = hessian_wrt_beta(mu, phi, X_f, Y).reshape((self.K*self.J,self.K*self.J))
            
        if self.spatial:
            hess_spatial = np.zeros((hess.shape[0]+1,hess.shape[1]+1))
            hess_spatial[:-1,:-1] = hess
            hess_spatial[-1,-1] = second_derivative_wrt_rho(mu, phi, beta, M, W, X_f, Y)
            der_rho_beta = second_derivative_wrt_rho_beta(mu, phi, beta, M, W, X_f, Y).flatten()
            hess_spatial[-1,:self.K*self.J] = der_rho_beta
            hess_spatial[:self.K*self.J,-1] = der_rho_beta
            if hasattr(self,'gamma'):
                der_rho_gamma = second_derivative_wrt_rho_gamma(mu, phi, beta, M, W, X_f, Y, Z)
                hess_spatial[-1,self.K*self.J:-1] = der_rho_gamma
                hess_spatial[self.K*self.J:-1,-1] = der_rho_gamma
            self.hess = hess_spatial
        else:
            self.hess = hess
                
        
    def pred(self, X, W=None):
        beta = np.zeros((self.K, self.J))
        beta[:,1:] = self.beta
        if self.intercept:
            X_f = np.ones((len(X),self.K))
            X_f[:,1:] = X
        else:
            X_f = X
        if self.spatial:
            M = np.identity(X.shape[0]) - self.rho*W
            mu = compute_mu_spatial(X_f, beta, M)
        else:
            mu = compute_mu(X_f, beta)
        return mu
        