import numpy as np
import gmm_module

filename = "test/data1.csv"
data = gmm_module.read_data(filename)
print("Read Data:", data)
print("Shape:", data.shape)

K = 3  
D = 2  
N = 100  
threshold = 0.001

data = np.random.rand(N * D).astype(np.float32)
pi_k = np.ones(K, dtype=np.float32) / K
u_k = np.random.rand(K * D).astype(np.float32)
E_k = np.tile(np.eye(D, dtype=np.float32), (K, 1)).flatten()

gmm_module.gmm_training(pi_k, u_k, E_k, data, K, D, N, threshold)

labels = gmm_module.gmm_inference(pi_k, u_k, E_k, data, K, D, N)
print("Cluster Labels:", labels)
