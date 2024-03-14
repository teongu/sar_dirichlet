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
