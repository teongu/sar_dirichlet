import numpy as np
import pandas as pd
from smote_cd.dataset_generation import softmax
from scipy.special import gamma, digamma, polygamma, loggamma
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from scipy.optimize import minimize, Bounds


def compute_mu(X, beta, Xbeta=None):
    n = np.shape(X)[0]
    J = np.shape(beta)[1] #number of classes
    if Xbeta is None:
        Xbeta = np.matmul(X,beta)
    mu = np.zeros((n,J))
    # compute the softmax
    exp_Xbeta = np.exp(Xbeta)
    exp_Xbeta[Xbeta>700] = 1e305 #we replace the values that overflow
    sum_exp_Xbeta = np.sum(exp_Xbeta,axis=1)
    mu = exp_Xbeta/sum_exp_Xbeta[:,None]
    mu[mu==0]=1e-305 #we replace the 0s
    return mu


def compute_mu_spatial(X, beta, M, Xbeta=None, MinvX=None, MXbeta=None):
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
    for i in range(n):
        gradient += np.outer(X[i],mu_phi_times_sum_terms[i])
    return gradient


def gradient_wrt_gamma(mu, phi, Y, Z, epsilon=0):
    mu_logY_minus_digamma_phi_mu = mu * ( np.log(Y+epsilon) - digamma(phi.reshape(-1,1)*mu) )
    phi_times_digamma_phi_plus_sum = phi * ( digamma(phi) + np.sum(mu_logY_minus_digamma_phi_mu, axis=1) )
    derivative_gamma = np.sum(Z * phi_times_digamma_phi_plus_sum.reshape(-1,1), axis=0)
    return(derivative_gamma)


def derivative_wrt_rho(mu, phi, beta, M, W, X, Y, Z, MinvX=None, MinvXbeta=None, alpha=None, epsilon=0):
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

def second_derivative_beta_gamma_2(mu, phi, beta, X, Y, Z, epsilon=0):
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


def second_derivative_wrt_rho(mu, phi, Minv, beta, W, X, Y, Z, MinvX=None, alpha=None, epsilon=0):
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


def second_derivative_wrt_rho_gamma(mu, phi, Minv, beta, W, X, Y, Z, MinvX=None, alpha=None, epsilon=0):
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


def second_derivative_wrt_rho_beta(mu, phi, Minv, beta, W, X, Y, Z, MinvX=None, alpha=None, epsilon=0):
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


####################

def objective_func_loglik_no_spatial(x, X, Y, Z=None, phi=None, epsilon=0, regularization_lambda=0):
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
        obj = obj + regularization_lambda * np.linalg.norm(x)
    return obj

def jac_objective_func_loglik_no_spatial(x, X, Y, Z=None, phi=None, epsilon=0, regularization_lambda=0):
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
        return -beta_grad.flatten()
    else:
        gamma_grad = gradient_wrt_gamma(mu, phi, Y, Z, epsilon=epsilon)
        return(-np.concatenate([beta_grad.flatten(),gamma_grad]))

def objective_func_loglik_spatial(x, X, Y, W, Z=None, phi=None, epsilon=0, regularization_lambda=0):
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
        obj = obj + regularization_lambda * np.linalg.norm(x)
    return obj


####################

class dirichletRegressor:
    
    def __init__(self, learning_rate=0.01, maxiter=1000, random_state=None, spatial=False, maxfun=1000):
        self.learning_rate = learning_rate
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.random_state = random_state
        self.spatial = spatial
        
    def fit(self, X , Y, W=None, beta_0=None, rho_0=0, parametrization='common', gamma_0=None, Z=None, fit_intercept=True):
        
        n,K = np.shape(X)
        J = Y.shape[1]
        if fit_intercept:
            K+=1
            self.intercept = True
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
            if parametrization=='common':
                params0 = np.concatenate([beta_0.flatten(),[rho_0]])
                min_bounds, max_bounds = -np.inf*np.ones(len(params0)), np.inf*np.ones(len(params0))
                min_bounds[-1], max_bounds[-1] = -1, 1
                bounds = Bounds(min_bounds, max_bounds)
                solution = minimize(objective_func_loglik_spatial, params0, args=(X_f, Y, W, None, self.phi), bounds=bounds, options={'maxiter':self.maxiter, 'maxfun':self.maxfun})
                self.beta = solution.x[:-1].reshape((K,J-1))
                
            else:
                params0 = np.concatenate([beta_0.flatten(),gamma_0,[rho_0]])
                min_bounds, max_bounds = -np.inf*np.ones(len(params0)), np.inf*np.ones(len(params0))
                min_bounds[-1], max_bounds[-1] = -1, 1
                bounds = Bounds(min_bounds, max_bounds)
                solution = minimize(objective_func_loglik_spatial, params0, args=(X_f, Y, W, Z), bounds=bounds, options={'maxiter':self.maxiter, 'maxfun':self.maxfun})
                self.beta = solution.x[:K*(J-1)].reshape((K,J-1))
                self.gamma = solution.x[K*(J-1):-1]
                self.phi = np.exp(np.matmul(Z,self.gamma))
            self.rho = solution.x[-1]
            beta = np.zeros((K, J))
            beta[:,1:] = self.beta
            M = np.identity(n) - self.rho*W
            self.mu = compute_mu_spatial(X_f, beta, M)
            
        else: 
            if parametrization=='common':
                params0 = beta_0.flatten()
                #solution = minimize(objective_func_loglik_no_spatial, params0, args=(X_f,Y,None,self.phi))
                solution = minimize(objective_func_loglik_no_spatial, params0, args=(X_f,Y,None,self.phi), jac = jac_objective_func_loglik_no_spatial)
                self.beta = solution.x.reshape((K,J-1))
            else:
                params0 = np.concatenate([beta_0.flatten(),gamma_0])
                solution = minimize(objective_func_loglik_no_spatial, params0, args=(X_f,Y,Z))
                self.beta = solution.x[:K*(J-1)].reshape((K,J-1))
                self.gamma = solution.x[K*(J-1):]
                self.phi = np.exp(np.matmul(Z,self.gamma))
                
            beta = np.zeros((K, J))
            beta[:,1:] = self.beta
            self.mu = compute_mu(X_f, beta)
                
        
        print(solution.message)
        
    def compute_mu(self, X, W=None, return_mu=False):
        beta = np.zeros((self.K, self.J))
        beta[:,1:] = self.beta
        if self.intercept:
            X_f = np.ones((len(X),self.K))
            X_f[:,1:] = X
        if self.spatial:
            M = np.identity(X.shape[0]) - self.rho*W
            self.mu = compute_mu_spatial(X_f, beta, M)
        else:
            self.mu = compute_mu(X_f, beta)
        if return_mu:
            return self.mu
        