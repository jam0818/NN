#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <string.h>
#include "nn.h"

void print(int m, int n, const float * x) {
    for(int i = 0; i < m * n; i += n){
        for(int j = i; j < i + n; j++){
            printf("%.4f ", x[j]);
        }
        printf("\n");
    }
}

void add(int n, const float * x, float * o) {
    for(int i = 0; i < n; i++){
        o[i] = x[i] + o[i];
    }
}

void scale(int n, float x, float * o) {
    for (int i = 0; i < n; i++) {
        o[i] *= x;
    }    
}

float GetRandom(float min,float max){
	return min + (float)(rand()*(max-min)/(1.0+RAND_MAX));
}

void rand_init(int n, float * o) {
    srand((unsigned)time(NULL));
    for(int i = 0; i < n; i++) {
        o[i] = GetRandom(-1, 1);
    }
}

void fc(int m, int n, const float * x, const float * A, const float * b, float *y) {
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            y[i] = y[i] + A[j + i * n] * x[j];
        }
        y[i] = y[i] + b[i];
    }
}

void relu(int n, const float * x, float * y) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0) {
            y[i] = 0;
        } else {
            y[i] = x[i];
        }
    }
}

void softmax(int n, const float * x, float * y) {
    float max;
    for (int i = 0; i < n; i++){
        if (max < x[i]){
            max = x[i];
        }
    }
    for (int i = 0; i < n; i++){
        float sum = 0;
        for (int j = 0; j < n; j++) {
            sum += exp(x[i] - max);
        }
        y[i] = (x[i] - max) / sum; 
    }
}


// テスト
int main() {    

    return 0;
}