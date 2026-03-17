import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors


def create_features_matrices(n_samples,n_features,choice_W='X_dependent',threshold_neighbors=0.3,nneighbors=5,cov_mat=None):
    if cov_mat is None:
        cov_mat = np.array([[1., 0.2], [0.2, 1.]])
    X = np.random.multivariate_normal([0.]*n_features,cov_mat,size=n_samples)
    X = np.array([np.concatenate(([1],x)) for x in X])
    Z = np.random.uniform(size=(n_samples,n_features))
    
    if choice_W == 'X_dependent':
        distance_matrix = scipy.spatial.distance_matrix(X,X)
        W = np.zeros(np.shape(distance_matrix))
        W[distance_matrix < threshold_neighbors] = 1
    elif choice_W == 'random_distance':
        random_spatial_distance = np.random.rand(n_samples,n_features)
        neighbors = NearestNeighbors(n_neighbors=nneighbors).fit(random_spatial_distance)
        W = neighbors.kneighbors_graph(random_spatial_distance, mode='distance').toarray()
        W[W>0] = 1/W[W>0]
    else:
        random_spatial_distance = np.random.rand(n_samples,n_features)
        neighbors = NearestNeighbors(n_neighbors=nneighbors).fit(random_spatial_distance)
        W = neighbors.kneighbors_graph(random_spatial_distance).toarray()
    # replace the 1 on the diagonal by 0
    np.fill_diagonal(W,0)
    # scaling the matrix, so that the sum of each row is 1
    W = W/W.sum(axis=1)[:,None]
    return(X,Z,W)


def cos_similarity(x1,x2):
    """
    Calculate the cosine similarity between two matrices.

    Cosine similarity measures the cosine of the angle between two vectors, providing a
    similarity value in the range [-1, 1], where higher values indicate greater similarity.

    Parameters:
    x1 (numpy.ndarray): The first matrix of shape (n, m).
    x2 (numpy.ndarray): The second matrix of shape (n, m), where n is the number of rows
                        and m is the number of features (columns).

    Returns:
    float: The mean cosine similarity between rows of x1 and x2.
    """
    return(np.mean([np.dot(x1[i],x2[i])/(np.linalg.norm(x1[i])*np.linalg.norm(x2[i])) for i in range(len(x1))]))



def rmse_aitchison(x1, x2):
    """
    Compute the mean RMSE under Aitchison geometry between rows of matrices x1 and x2.
    
    Parameters:
    x1 (numpy.ndarray): The first matrix of shape (n, m).
    x2 (numpy.ndarray): The second matrix of shape (n, m), where n is the number of rows
                        and m is the number of features (columns).
    
    Returns:
    float: The mean RMSE_A across all rows.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape.")
    if np.any(x1 <= 0) or np.any(x2 <= 0):
        raise ValueError("All components must be strictly positive.")

    n, D = x1.shape
    log_x1 = np.log(x1)
    log_x2 = np.log(x2)

    # Compute log-ratio matrices: log(x_i) - log(x_j) for each row
    diff_x1 = log_x1[:, :, None] - log_x1[:, None, :]
    diff_x2 = log_x2[:, :, None] - log_x2[:, None, :]
    
    # Difference of differences: shape (n, D, D)
    delta = diff_x1 - diff_x2
    squared = delta ** 2

    # Sum over i and j (axes 1 and 2), then average over rows
    row_rmse = np.sqrt(np.sum(squared, axis=(1, 2)) / (2 * D))
    return np.mean(row_rmse)



def aitchison_mean(y):
    """
    Compute the Aitchison (compositional) mean of compositional data.
    
    Parameters:
    y (numpy.ndarray): Matrix of shape (n, J) where each row is a composition
    
    Returns:
    numpy.ndarray: The Aitchison mean composition (length J)
    """
    # Geometric mean of each component across samples
    geometric_means = np.exp(np.mean(np.log(y), axis=0))
    # Closure operation (normalize to sum to 1)
    return geometric_means / np.sum(geometric_means)


def aitchison_inner_product(x1, x2):
    """
    Compute the Aitchison inner product between two compositions.
    
    Parameters:
    x1, x2 (numpy.ndarray): Compositional vectors of length J
    
    Returns:
    float: The Aitchison inner product ⟨x1, x2⟩_A
    """
    J = len(x1)
    log_x1 = np.log(x1)
    log_x2 = np.log(x2)
    
    # Compute the double sum over j and j'
    inner_sum = 0
    for j in range(J):
        for jp in range(J):
            log_ratio_x1 = np.log(x1[j] / x1[jp])
            log_ratio_x2 = np.log(x2[j] / x2[jp])
            inner_sum += log_ratio_x1 * log_ratio_x2
    
    # Divide by 2J as per the convention in the paper
    return inner_sum / (2 * J)


def aitchison_norm(x):
    """
    Compute the Aitchison norm of a composition.
    
    Parameters:
    x (numpy.ndarray): Compositional vector of length J
    
    Returns:
    float: The Aitchison norm ‖x‖_A
    """
    return np.sqrt(aitchison_inner_product(x, x))


def aitchison_distance(x1, x2):
    """
    Compute the Aitchison distance between two compositions.
    
    Parameters:
    x1, x2 (numpy.ndarray): Compositional vectors of length J
    
    Returns:
    float: The Aitchison distance d_A(x1, x2)
    """
    J = len(x1)
    log_x1 = np.log(x1)
    log_x2 = np.log(x2)
    
    # Compute the double sum over j and j'
    inner_sum = 0
    for j in range(J):
        for jp in range(J):
            log_ratio_x1 = np.log(x1[j] / x1[jp])
            log_ratio_x2 = np.log(x2[j] / x2[jp])
            inner_sum += (log_ratio_x1 - log_ratio_x2) ** 2
    
    # Divide by 2J and take square root
    return np.sqrt(inner_sum / (2 * J))


def r2_aitchison(y_true, y_pred):
    """
    Compute the Aitchison-based R² coefficient of determination.
    
    R_A² = 1 - RSS_A / TSS_A
    
    where:
    - RSS_A = Σ d_A(y_i, ŷ_i)² (residual sum of squares)
    - TSS_A = Σ d_A(y_i, ȳ_A)² (total sum of squares)
    - ȳ_A is the Aitchison mean of the observations
    
    Parameters:
    y_true (numpy.ndarray): True compositions, shape (n, J)
    y_pred (numpy.ndarray): Predicted compositions, shape (n, J)
    
    Returns:
    float: The Aitchison R² value
    """
    # Compute Aitchison mean of true values
    y_mean = aitchison_mean(y_true)
    
    # Compute total sum of squares (TSS_A)
    tss = 0
    for i in range(len(y_true)):
        tss += aitchison_distance(y_true[i], y_mean) ** 2
    
    # Compute residual sum of squares (RSS_A)
    rss = 0
    for i in range(len(y_true)):
        rss += aitchison_distance(y_true[i], y_pred[i]) ** 2
    
    # Compute R²
    r2 = 1 - (rss / tss) if tss > 0 else 0
    return r2


def r2_aitchison_adjusted(y_true, y_pred, n_params):
    """
    Compute the adjusted Aitchison-based R².
    
    R_A,adj² = 1 - (1 - R_A²) * (n - 1) / (n - k)
    
    where:
    - n is the number of samples
    - k is the number of estimated parameters
    
    Parameters:
    y_true (numpy.ndarray): True compositions, shape (n, J)
    y_pred (numpy.ndarray): Predicted compositions, shape (n, J)
    n_params (int): Number of estimated parameters in the model
    
    Returns:
    float: The adjusted Aitchison R² value
    """
    n = len(y_true)
    r2 = r2_aitchison(y_true, y_pred)
    
    # Handle edge cases
    if n <= n_params or n <= 1:
        return r2
    
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - n_params)
    return r2_adj


def cosine_similarity_aitchison(y_true, y_pred):
    """
    Compute the Aitchison-based cosine similarity.
    
    Cosine_A = (1/n) Σ ⟨y_i, ŷ_i⟩_A / (‖y_i‖_A * ‖ŷ_i‖_A)
    
    Parameters:
    y_true (numpy.ndarray): True compositions, shape (n, J)
    y_pred (numpy.ndarray): Predicted compositions, shape (n, J)
    
    Returns:
    float: The mean Aitchison cosine similarity across all samples
    """
    n = len(y_true)
    total_cosine = 0
    
    for i in range(n):
        inner_prod = aitchison_inner_product(y_true[i], y_pred[i])
        norm_true = aitchison_norm(y_true[i])
        norm_pred = aitchison_norm(y_pred[i])
        
        # Avoid division by zero
        if norm_true > 0 and norm_pred > 0:
            cosine = inner_prod / (norm_true * norm_pred)
        else:
            cosine = 0
        
        total_cosine += cosine
    
    return total_cosine / n