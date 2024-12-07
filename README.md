# TO BUILD PROGRAM RUN
bash make.bash

# TO CLEAN BUILD FILES RUN
bash make_clean.bash

# running benchmark.py
flags:  
-n : number of data points  
-d : data dimension  
-k : number of clusters | higher k higher benchmark  
-t : threshold for EM convergence  
-mi : max number of iterations allowed(optional parameter | default:500)  
-m : data generation mode | 0 : generate blob like data | 1 : generate random dense data(tougher benchmark)  
-v : verbose mode | print iterations  
-b : bash mode | prints GPU_time, Iters/s, SEQ_time, Iters/s | used for automated bash testing  
-g : run GPU mode only(if seq version is too slow)  
-p : plot data in res.png(disable for efficiency)  


# sample usage
python3 benchmark.py -n 10000 -d 2 -k 4 -t 0.0001 -mi 100 -m 1 -v -p



