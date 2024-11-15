#include <iostream>
#include <vector>

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

}