#include <iostream>
#include <vector>

using namespace std;

#include "csv_read.hpp"

//initial params means(u_k),variances (E_k) and mixing(pi_k)
//D dim data | N points | K clusters

/*
Params
pi_k = [*]xK
u_k  = [*,*]x(K,D)      | access: (k*D)+i
E_k  = [*,*,*]x(K,D,D)  | access: k*(D*D) +(i*D) + j


Data point:
Dataset = [*,*]x(N,D)   | access: i*(D) + j


*/

int main() {

	int K=5; int D=2; int N=200;
	float pi_k[K];
	float u_k[(K*D)];
	float E_k[(K*D*D)];

	read_data(pi_k,"weights.csv");
	read_data(u_k,"means.csv");
	read_data(E_k,"covariances.csv");

	for (int i=0;i<K*D;i++) {
		cout << u_k[i] << endl;
	}

}