from class_def import GMM
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import time
import sys
import argparse

original_write = sys.stdout.write
prev_line_length=0
def custom_write(text):
    global prev_line_length
    # Strip the text of trailing newlines
    stripped_text = text.strip()
    
    # If the line is not empty
    if stripped_text:
        # Clear the previous line by overwriting it with spaces
        clear_line = "\r" + " " * prev_line_length + "\r"
        original_write(clear_line)  # Clear the previous line
        
        # Write the new line and flush
        original_write(stripped_text)
        sys.stdout.flush()
        
        # Update the previous line length
        prev_line_length = len(stripped_text)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', type=int, help='data mode', required=True)
parser.add_argument('--data_size','-n',type=int, help='# of data points', required=True)
parser.add_argument('--dimension','-d',type=int,help='# data dimension',required=True)
parser.add_argument('--n_clusters','-k',type=int,help='# clusters',required=True)
parser.add_argument('--threshold','-t',type=float,help='# threshold',required=True)
parser.add_argument('--max_iterations','-mi',type=int,help='# max no. of iterations',required=False)
parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose mode')
parser.add_argument('--bash', '-b', action='store_true', help='Enable bash mode')
parser.add_argument('--plot', '-p', action='store_true', help='Enable plot mode')
parser.add_argument('--gpu_only', '-g', action='store_true', help='enable gpu only')
parser.add_argument('--infer', '-i', action='store_true', help='inference')
bash_out = ""

args = parser.parse_args()

N = args.data_size   # number of points
D = args.dimension         # dimension
K = args.n_clusters         # clusters
T = args.threshold    # threshold
M = args.mode         # running mode
MAX_ITER = 500
if (args.max_iterations):
    MAX_ITER = args.max_iterations  #max iterations
V = args.verbose        #enable verbose mode
B = args.bash           # bash output mode
P = args.plot          #plot data
G = args.gpu_only      #gpu mode
I = args.infer         #enable inference

start_time=None
end_time=None

#read data
#data = pd.read_csv('data.csv', header=None)
data = None
#use for clustered data
if M==0:
    data,_ = make_blobs(n_samples=N,centers=np.random.rand(K, D) * 200, n_features=D,cluster_std=8.0)

#use for random data(heavy benchmark)
if M==1:
    data = np.random.rand(N, D) * 1000

model = GMM(data,K,T) #args: # clusters and threshold
start_time = time.time()
print("\nTraining on GPU...")
model.train(V,MAX_ITER)
end_time = time.time()
gpu_iterations = model.n_iter_
if (B):
    bash_out += f'{end_time-start_time},{gpu_iterations/(end_time-start_time):0.3f}'
else:
    print(f'GPU TIME: {(end_time-start_time):0.4f} || Iters/s: {gpu_iterations/(end_time-start_time):0.3f}')

gmm = None
if (V):
    gmm = GaussianMixture(n_components=K,tol=T, init_params='kmeans',covariance_type='full',verbose=2,verbose_interval=1)
else:
    gmm = GaussianMixture(n_components=K,tol=T, init_params='kmeans',covariance_type='full')
seq_iterations = 0
if not G:
    start_time = time.time()
    print("\nTraining on CPU...")
    sys.stdout.write = custom_write
    gmm.fit(data)
    sys.stdout.write = original_write
    end_time = time.time()
    seq_iterations = gmm.n_iter_
if (B):
    if not G:
        bash_out += ','+f'{end_time-start_time},{gpu_iterations/(end_time-start_time):0.3f}'
else:
    if not G:
        print(f'\nSEQ TIME: {(end_time-start_time):0.4f} || Iters/s: {seq_iterations/(end_time-start_time):0.3f}')


labels = None
if I:
    start_time = time.time()
    print("\nInferece on GPU...")
    labels = model.predict(data)
    end_time = time.time()
    if B:
        bash_out = ""
        bash_out+=f'{end_time-start_time},'
    else:
        print(f'GPU TIME Inf: {(end_time-start_time):0.4f}')
    if not G:
        start_time = time.time()
        print("\nInferece on CPU...")
        labels1 = gmm.predict(data)
        end_time = time.time()
        if B:
            bash_out+=f'{end_time-start_time}'
        else:
            print(f'CPU TIME Inf: {(end_time-start_time):0.4f}')

if B:
    print(bash_out)

#plot data
if P:
    print("\nPlotting...",end="")
    start_time = time.time()
    labels = model.predict(data)
    means = model.means()
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:,0], data[:,1], c=labels, cmap='viridis', marker='o')
    plt.scatter(means[:,0], means[:,1], c='red', marker='x', s=100, label='Centroids')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaussian Mixture Model Clustering')
    plt.savefig("res.png")
    end_time = time.time()
    print(f'\rPlotting completed in {(end_time-start_time):0.1f}s')

#save data for reproducing
#df = pd.DataFrame(data)
#df.to_csv('data.csv', index=False, header=False)
