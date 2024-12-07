import numpy as np
from sklearn.cluster import KMeans
import gmm_module

class GMM:
    def __init__(self,data,n_clusters,threshold):
        self.data = data
        self.K = n_clusters
        self.N = data.shape[0]
        self.D = data.shape[1]
        self.threshold = threshold
        self.n_iter_ = 0
        self.res_labels = []

    def train(self,verbose,max_iter=500):
        kmeans = KMeans(n_clusters=self.K, random_state=42,n_init='auto')
        labels = kmeans.fit_predict(self.data)
        pi_k = (np.bincount(labels) / self.N).flatten().astype(np.double)
        u_k = (kmeans.cluster_centers_).flatten().astype(np.double)
        covariances = []
        for k in range(self.K):
            # Extract data points belonging to the k-th cluster
            cluster_data = self.data[labels == k]
            # Compute covariance matrix
            cov_matrix = np.cov(cluster_data, rowvar=False)
            covariances.append(cov_matrix)

        E_k = (np.vstack(covariances)).flatten().astype(np.double)
        self.n_iter_ = gmm_module.gmm_training(pi_k, u_k, E_k, self.data, self.K, self.D, self.N, self.threshold,verbose,max_iter)
        self.pi_k = pi_k
        self.u_k = u_k
        self.E_k = E_k
        self.res_labels = gmm_module.gmm_inference(pi_k, u_k, E_k,self.data.flatten().astype(np.double),self.K, self.D, self.N)

    def predict(self,input_data):
        return gmm_module.gmm_inference(self.pi_k, self.u_k, self.E_k, input_data.flatten().astype(np.double), self.K, self.D, input_data.shape[0])

    def means(self):
        return self.u_k.reshape(self.K,self.D)

