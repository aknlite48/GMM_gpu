#include <iostream>
#include <vector>

# define M_PI 3.14159265358979323846  /* pi */

using namespace std;

#include "csv_read.hpp"
#include "kernels.h"

//initial params means(u_k),variances (E_k) and mixing(pi_k)
//D dim data | N points | K clusters



//Data point:
//Dataset = [*,*]x(N,D)   | access: i
int main(int argc,char* argv[]) {
	if (argc!=13) {
		cout << "Incorrect Usage => -K num_clusters -D data_dim -N data_size -T threshold -S data_size -F filename" << endl;
		return 1;
	}

	int K, D,  N; float threshold;
       	int inf_size; string filename; //inference

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "-N") {
            if (i + 1 < argc) { // Check if there is a value after the flag
                N = std::stoi(argv[++i]); // Parse the next argument as an integer

            } 
        } else if (arg == "-K") {
                        if (i + 1 < argc) { // Check if there is a value after the flag
                K = std::stoi(argv[++i]); // Parse the next argument as an integer

            } 
        } else if (arg=="-D") {
        	            if (i + 1 < argc) { // Check if there is a value after the flag
                D = std::stoi(argv[++i]); // Parse the next argument as an integer

            } 
        } else if (arg=="-T") {
                            if (i + 1 < argc) { // Check if there is a value after the flag
                threshold = std::stof(argv[++i]); // Parse the next argument as an integer
			    }
        } else if (arg=="-S") {
                            if (i + 1 < argc) { // Check if there is a value after the flag
                inf_size = std::stof(argv[++i]); // Parse the next argument as an integer
			    }
        } else if (arg=="-F") {
                            if (i + 1 < argc) { // Check if there is a value after the flag
                filename = argv[++i]; // Parse the next argument as an integer

            }
} 
    }

    float pi_k[K];
	float u_k[(K*D)];
	float E_k[(K*D*D)];
	float data[(N*D)];
	read_data(data,"data.csv");
	read_data(pi_k,"weights.csv");
	read_data(u_k,"means.csv");
	read_data(E_k,"covariances.csv");

	//training:
	GMM_training(pi_k,u_k,E_k,data,K,D,N,threshold);






    //inference 
    float  data_inf[(D*inf_size)];
    int labels[inf_size];
    read_data(data_inf,filename);
    GMM_inference(labels,pi_k,u_k,E_k,data_inf,K,D,inf_size);

	for (int i=0;i<inf_size;i++) {
		printf("%d : %d \n",i,labels[i]);
	}

}
