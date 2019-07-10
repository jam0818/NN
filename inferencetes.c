#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
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

//行列の表示
void print(int m, int n, const float * x) {
    for(int i = 0; i < m ; i++){
        for(int j = 0; j < n; j++){
            printf("%.4f ", x[j + i * n]);
        }
        printf("\n");
    }
}

//学習した係数の読み取り
void load(const char * filename, int m, int n, float * A, float * b) {
    
    FILE *fp;
    if((fp = fopen(filename,"rb"))==NULL){
        printf("\aファイルをオープンできません。\n");
    } else {
        fread(A, sizeof(float), m * n, fp);
        fread(b, sizeof(float), n, fp);
        fclose(fp);
    }
}

void fc(int m,
        int n,
        const float *x,  // (n,)
        const float *A,  // (m, n)
        const float *b,  // (m,)
        float *y         // (m,)
        ) {
    #pragma omp parallel for
    for(int i = 0; i < m; i++){
        y[i] = 0;
        #pragma omp parallel for
        for(int j = 0; j < n; j++){
            y[i] = y[i] + A[j + i * n] * x[j];
        }
        y[i] = y[i] + b[i];
    }
}


//relu層（順伝播）
void relu(int n, const float * x, float * y) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (x[i] < 0) {
            y[i] = 0;
        } else {
            y[i] = x[i];
        }
    }
}

//softmax層（順伝播）
void softmax(int n, const float * x, float * y) {
    float max = 0;
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        if (max < x[i]){
            max = x[i];
        }
    }
    float sum = 0;
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        sum += exp(x[i] - max);
    }
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        y[i] = (exp(x[i] - max) / sum);
    }
}

int inference6(const float*A1,
               const float *b1,
               const float*A3,
               const float*b3,
               const float*A5,
               const float*b5, 
               const float *x, 
               float *y6
               ){
    float *y1 = malloc(sizeof(float) * 50); // (50,)
    float *y2 = malloc(sizeof(float) * 50); // (50,)
    float *y3 = malloc(sizeof(float) * 100); // (100,)
    float *y4 = malloc(sizeof(float) * 100); // (100,)
    float *y5 = malloc(sizeof(float) * 10); // (50,)
    //順伝播は左端の引数が出力値
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y2);
    fc(100, 50, y2, A3, b3, y3);
    relu(100, y3, y4);
    fc(10, 100, y4, A5, b5, y5);
    softmax(10, y5, y6);
    int temp = 1;
    float M = 0;
    for (int i = 0; i < 10; i++){
        if (M < y6[i]){
            M = y6[i];
        }
    }
    for (int i = 0; i < 10; i++){
        if (M == y6[i])
        temp = i;
    }     
    free(y1);
    free(y2);
    free(y3);
    free(y4);
    free(y5);
    return temp;

}

//指定した値で初期化
void init(int n, float x, float * o) {
    for(int i = 0; i < n; i++) {
        o[i] = x;
    }
}

int main(int argc, char const *argv[]) {
    float * train_x = NULL;
    unsigned char * train_y = NULL;
    int train_count = -1;
    float * test_x = NULL;
    unsigned char * test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count,
    &test_x, &test_y, &test_count,
    &width, &height);
    
// これ以降，３層 NN の係数 A_784x10 および b_784x10 と，
// 訓練データ train_x + 784*i (i=0,...,train_count-1), train_y[0]～train_y[train_count-1],
// テストデータ test_x + 784*i (i=0,...,test_count-1), test_y[0]～test_y[test_count-1],
// を使用することができる．
    float * A1 = malloc(sizeof(float)*784*50);
    float * b1 = malloc(sizeof(float)*50);
    float * A3 = malloc(sizeof(float)*50*100);
    float * b3 = malloc(sizeof(float)*100);
    float * A5 = malloc(sizeof(float)*100*10);
    float * b5 = malloc(sizeof(float)*10);
    float * y6 = malloc(sizeof(float)*10);
    init(784*50,0,A1);
    init(50*100,0,A3);
    init(100*10,0,A5);
    init(50,0,b1);
    init(100,0,b3);
    init(10,0,b5);
    init(10,0,y6);
            int sum_train = 0;
            float loss_train = 0;
            float acc_train = 0;
            int sum_test = 0;
            float acc_test = 0;
            float loss_test = 0;
    load(argv[1], 784, 50, A1, b1);

    load(argv[2], 50, 100, A3, b3);

    load(argv[3], 100, 10, A5, b5);

            for (int k = 0; k < test_count; k++) {
                if (inference6(A1, b1, A3, b3, A5, b5, test_x + 784 * k, y6) == test_y[k]) {
                    sum_test++;
                }
            }
            acc_test = sum_test * 100.0 / test_count;
            printf("\naccuracy(test) : %f%%\n", acc_test);

    return 0;
}