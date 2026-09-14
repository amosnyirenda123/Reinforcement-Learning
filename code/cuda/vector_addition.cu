#include<stdio.h>
#include<stdlib.h>

#define N 512


void host_add(int *a, int *b, int *c){
    for (size_t i = 0; i < N; i++)
        c[i] = a[i] + b[i];
}

void fill_array(int *data){
    for (size_t i = 0; i < N; i++)
        data[i] = i;
}

void print_output(int *a, int *b, int*c) {
	for(int idx=0;idx<N;idx++)
		printf("\n %d + %d  = %d",  a[idx] , b[idx], c[idx]);
}


int main(void){
    int *a, *b, *c;

    int size = N * sizeof(int);

    a = (int*) malloc(size); fill_array(a);
    b = (int*) malloc(size); fill_array(b);
    c = (int*) malloc(size); fill_array(c);

    host_add(a, b, c);
    print_output(a, b, c);

    free(a); free(b); free(c);

    return EXIT_SUCCESS;
}