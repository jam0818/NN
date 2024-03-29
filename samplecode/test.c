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

//スワップ関数int
void swapi(int *pa, int *pb){
    int temp = *pa;
    *pa = *pb;
    *pb = temp;
}

//スワップ関数float
void swap(float *pa, float *pb){
    float temp = *pa;
    *pa = *pb;
    *pb = temp;
}

//ベクトルの足し算
void add(int n, const float * x, float * o) {
    for(int i = 0; i < n; i++){
        o[i] = x[i] + o[i];
    }
}

//指定した値で初期化
void init(int n, float x, float * o) {
    for(int i = 0; i < n; i++) {
        o[i] = x;
    }
}

//行列の表示
void print(int m, int n, const float * x) {
    for(int i = 0; i < m ; i++){
        for(int j = 0; j < n; j++){
            printf("%.4f ", x[j + i * n]);
        }
        printf("\n");
    }
}

//スカラー倍
void scale(int n, float x, float * o) {
    for (int i = 0; i < n; i++) {
        o[i] *= x;
    }    
}

//一定の範囲での乱数生成関数
float GetRandom(float min,float max){
	return min + (float)(rand()*(max-min)/(1.0+RAND_MAX));
}

//初期化関数
void rand_init(int n, float * o) {
    srand((unsigned)time(NULL));
    for(int i = 0; i < n; i++) {
        o[i] = GetRandom(-1, 1);
    }
}

//fc層（順伝播）
void fc(int m, int n, const float * x, const float * A, const float * b, float *y) {
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            y[i] = y[i] + A[j + i * n] * x[j];
        }
        y[i] = y[i] + b[i];
    }
}

//relu層（順伝播）
void relu(int n, const float * x, float * y) {
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
    for (int i = 0; i < n; i++){
        if (max < x[i]){
            max = x[i];
        }
    }
    float sum = 0;
    for (int i = 0; i < n; i++){
        sum += exp(x[i] - max);
    }
    for (int i = 0; i < n; i++){
        y[i] = (exp(x[i] - max) / sum);
    }
}

//推論（三層）
int inference3(const float * A, const float * b, const float * x) {
    float *y = malloc(sizeof(float) * 10);

    fc(10, 784, x, A, b, y);

    relu(10, y, y);

    softmax(10, y, y);

    int temp = 1;
    float M;
    for (int i = 0; i < 10; i++){
        if (M < y[i]){
            M = y[i];
        }
    }
    for (int i = 0; i < 10; i++){
        if (M == y[i])
        temp = i;
    }      
    free(y); 
    return temp;

}

//softmax層（逆伝播）
void softmaxwithloss_bwd(int n, const float * y, unsigned char t, float * dEdx) {
    for (int i = 0; i < n; i++) {
        if (i == t) {
            dEdx[i] = y[i] - 1;
        } else {
            dEdx[i] = y[i];
        }
    }
}

//Relu層（逆伝播）
void relu_bwd(int n, const float * x, const float * dEdy, float * dEdx) {
    for (int i = 0; i < n; i++) {
        if (x[i] > 0) {
            dEdx[i] = dEdy[i];
        } else {
            dEdx[i] = 0;
        }
    }
}

//fc層（逆伝播）
void fc_bwd(int m, int n, const float * x, const float * dEdy, const float * A, float * dEdA, float * dEdb, float * dEdx) {
    //dEdAの計算
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            dEdA[j + i * n] = dEdy[i] * x[j + i * n];
        }
    }
    //dEdbの計算
    for (int i = 0; i < m; i++) {
        dEdb[i] = dEdy[i];
    }
    //下流へ転送する勾配
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m ; j++){
            dEdx[i] += A[j * n + i] * dEdy[j];
        }
    }
}

//back prop(三層)
void backward3(const float *A, const float *b, const float *x, unsigned char t, float *y, float *dEdA, float *dEdb){

    float *temp_relu=malloc(sizeof(float)*10);
    fc(10, 784, x, A, b, y);

    relu(10, y, y);
    for (int i = 0; i < 10; i++) {
        temp_relu[i] = y[i];
    }

    softmax(10, y, y);

    float *dEdx=malloc(sizeof(float)*10);
    softmaxwithloss_bwd(10, y, t, dEdx);
    relu_bwd(10, temp_relu, dEdx, dEdx);
    float *dEdx784 = malloc(sizeof(float) * 784);


    fc_bwd(10, 784 , x, dEdx, A, dEdA, dEdb, dEdx784);
    free(temp_relu);
    free(dEdx);
    free(dEdx784);
}

//ランダムシャッフル
void shuffle(int n, int *x){
    srand(time(NULL));
    for (int i = 0; i < n;i++){
        int num = rand() % n;
        swapi(&x[i], &x[num]);
    }
}

//損失関数
float cross_entropy_error(const float * y, int t) {
    return -log(y[t] + 1e-7);
}

//学習した係数の保存
void save(const char *filename, int m, int n, const float *A, const float*b){
    
    FILE *fp;
    if((fp = fopen(filename,"wb"))==NULL){
        printf("\aファイルをオープンできません。\n");
    } else {
        fwrite(A, sizeof(float), m * n, fp);
        fwrite(b, sizeof(float), n, fp);
        fclose(fp);
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

//推論（6層）
int inference6(const float*A1,const float *b1,const float*A3,const float*b3,const float*A5,const float*b5, const float *x){
    float *y1 = malloc(sizeof(float) * 50);
    float *y2 = malloc(sizeof(float) * 50);
    float *y3 = malloc(sizeof(float) * 100);
    float *y4 = malloc(sizeof(float) * 100);
    float *y5 = malloc(sizeof(float) * 10);
    float *y6 = malloc(sizeof(float) * 10);
    //順伝播は左端の引数が出力値
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y2);
    fc(100, 50, y2, A3, b3, y3);
    relu(100, y3, y4);
    fc(10, 100, y4, A5, b5, y5);
    softmax(10, y5, y6);
    int temp = 1;
    float M;
    for (int i = 0; i < 10; i++){
        if (M < y6[i]){
            M = y6[i];
        }
    }
    for (int i = 0; i < 10; i++){
        if (M == y6[i])
        temp = i;
    }     

    return temp;

}

//back prop（6層）
void backward6(const float * A1, const float * b1, const float * A3, const float * b3, const float * A5, const float * b5, const float * x, unsigned char t, float *y1,float *y2, float *y3, float *y4, float *y5, float *y6, float *dA1, float *db1, float *dA3, float *db3,float *dA5, float *db5){


    //順伝播は左端の引数が出力値
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y2);
    fc(100, 50, y2, A3, b3, y3);
    relu(100, y3, y4);
    fc(10, 100, y4, A5, b5, y5);
    print(1,10,y5);
    softmax(10, y5, y6);
    print(1,10,y6);

    float *dx6 = malloc(sizeof(float) * 10);
    float *dx5 = malloc(sizeof(float) * 10);
    float *dx4 = malloc(sizeof(float) * 100);
    float *dx3 = malloc(sizeof(float) * 100);
    float *dx2 = malloc(sizeof(float) * 50);
    float *dx1 = malloc(sizeof(float) * 50);



    softmaxwithloss_bwd(10, y6, t, dx6);
    fc_bwd(10 , 100, x, dx6, A5, dA5, db5, dx5);
    relu_bwd(100, x, dx5, dx4);
    fc_bwd(100, 50, x, dx4, A3, dA3, db3, dx3);
    relu_bwd(10, x, dx3, dx2);
    fc_bwd(50, 784, x, dx2, A1, dA1, db1, dx1);
    
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

    float *y1 = malloc(sizeof(float) * 50);
    float *y2 = malloc(sizeof(float) * 50);
    float *y3 = malloc(sizeof(float) * 100);
    float *y4 = malloc(sizeof(float) * 100);
    float *y5 = malloc(sizeof(float) * 10);
    float *y6 = malloc(sizeof(float) * 10);
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *b1 = malloc(sizeof(float) * 50);
    float *A3 = malloc(sizeof(float) * 50 * 100);
    float *b3 = malloc(sizeof(float) * 100);
    float *A5 = malloc(sizeof(float) * 100 * 10);
    float *b5 = malloc(sizeof(float) * 10);
    float *dA5 = malloc(sizeof(float) * 100 * 10);
    float *dA3 = malloc(sizeof(float) * 50 * 100);
    float *dA1 = malloc(sizeof(float) * 784 * 50);
    float *db5 = malloc(sizeof(float) * 10);
    float *db3 = malloc(sizeof(float) * 100);
    float *db1 = malloc(sizeof(float) * 50);
    float *dx6 = malloc(sizeof(float) * 10);
    float *dx5 = malloc(sizeof(float) * 10);
    float *dx4 = malloc(sizeof(float) * 100);
    float *dx3 = malloc(sizeof(float) * 100);
    float *dx2 = malloc(sizeof(float) * 50);
    float *dx1 = malloc(sizeof(float) * 50);
    rand_init(784 * 50, A1);
    rand_init(50, b1);
    rand_init(50 * 100, A3);
    rand_init(100, b3);
    rand_init(100 * 10, A5);
    rand_init(10, b5);
    backward6(A1, b1, A3, b3, A5, b5, train_x, train_y[0], y1, y2, y3, y4, y5, y6, dA1, db1, dA3, db3, dA5, db5);
    
    print(1, 10, y6);

    return 0;
}