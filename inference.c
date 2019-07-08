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


//ベクトル化関数
void unroll(int m,
            int n,
            float matrix[m][n], //(m, n)
            float vector[m * n] //(m * n,)←ただしこれは(m, n)として扱う場合が多いかもしれない
            ) {
                for(int i = 0; i < m; i++) {
                    for(int j = 0; j < n; j++) {
                        vector[j + i * n] = matrix[i][j];
                    }
                }
}

//行列化関数
void makematrix(int m, 
                int n,
                float matrix[m][n], //(m, n)
                float vector[m * n] //(m * n,)←ただしこれは(m, n)として扱う場合が多いかもしれない
                ) {
                for(int i = 0; i < m; i++) {
                    for(int j = 0; j < n; j++) {
                        matrix[i][j] = vector[j + i * n];
                    }
                }
}

void makematrixc(int m, 
                int n,
                float matrix[m][n], //(m, n)
                const float vector[m * n] //(m * n,)←ただしこれは(m, n)として扱う場合が多いかもしれない
                ) {
                for(int i = 0; i < m; i++) {
                    for(int j = 0; j < n; j++) {
                        matrix[i][j] = vector[j + i * n];
                    }
                }
}

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
void add(int n, const float * x, float o[n]) {
    for(int i = 0; i < n; i++){
        o[i] = x[i] + o[i];
    }
}

void addmatrix (int m, int n, float x[m][n], float o[m][n]) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
        o[i][j] += x[i][j];            
        }
    }
}

//指定した値で初期化
void init(int n, float x, float o[n]) {
    for(int i = 0; i < n; i++) {
        o[i] = x;
    }
}

void initmatrix(int m, int n, float x, float o[m][n]) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
        o[i][j] = x;            
        }
    }
}

//行列の表示
void print(int m, int n, const float x[m * n]) {
    for(int i = 0; i < m ; i++){
        for(int j = 0; j < n; j++){
            printf("%.4f ", x[j + i * n]);
        }
        printf("\n");
    }
}

//スカラー倍
void scale(int n, float x, float o[n]) {
    for (int i = 0; i < n; i++) {
        o[i] *= x;
    }    
}

void scalematrix(int m, int n, float x, float o[m][n]) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
        o[i][j] *= x;            
        }
    }
}

//一定の範囲での乱数生成関数
float GetRandom(float min,float max){
	return min + (float)(rand()*(max-min)/(RAND_MAX));
}

//初期化関数
void rand_init(int n, float o[n]) {
    for(int i = 0; i < n; i++) {
        o[i] = GetRandom(-1, 1);
    }
}

void rand_initmatrix(int m, int n, float o[m][n]) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
        o[i][j] = GetRandom(-1, 1);            
        }
    }
}

//0 < x < 1の乱数
double Uniform( void ){
    return ((double)rand()+1.0)/((double)RAND_MAX+2.0);
}

//Box-muller法 正規分布乱数 mu:平均値 sigma:標準偏差
double rand_normal(double mu, double sigma){
    double z = sqrt( -2.0 * log(Uniform())) * sin( 2.0 * M_PI * Uniform());
    return mu + sigma*z;
}

//Heの正規分布による初期化
void he_init(int n, float o[n]){
    srand(time(NULL));
    for (int i = 0; i < n; i++){
        o[i] = rand_normal(0, sqrt(2.0/n));
    }
}

//fc層（順伝播）
void fc(int m,
        int n,
        const float x[], // (n,)
        const float A[], // (m, n)
        const float b[], // (m,)
        float y[] // (m,)
        ) {
    
    for(int i = 0; i < m; i++){
        y[i] = 0;
        
        for(int j = 0; j < n; j++){
            y[i] = y[i] + A[j + i * n] * x[j];
        }
        y[i] = y[i] + b[i];
    }
}


//relu層（順伝播）
void relu(int n, const float x[n], float y[n]) {
    
    for (int i = 0; i < n; i++) {
        if (x[i] < 0) {
            y[i] = 0;
        } else {
            y[i] = x[i];
        }
    }
}

void relumatrix1(const float X[][24], float Y[][24]) {
    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 24; j++) {
            if (X[i][j] < 0) {
                Y[i][j] = 0;
            } else {
                Y[i][j] = X[i][j];
            }
        }
    }
}

void relumatrix2(const float X[][8], float Y[][8]) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (X[i][j] < 0) {
                Y[i][j] = 0;
            } else {
                Y[i][j] = X[i][j];
            }
        }
    }
}


//softmax層（順伝播）
void softmax(int n, const float x[], float y[]) {
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

//畳み込み層（順伝播）
void convolution1 (
            const float W[5][5], //(m, n)
            const float b, //(M - m + 1, N - n + 1)
            const float X[28][28], //(M, N)
            float Y[24][24] //(M - m + 1, N - n + 1)
            ) {

    for (int i = 0; i < 24; i++) {
        for (int j = 0; j < 24; j++) {
            for (int s = 0; s < 5; s++) {
                for (int t = 0; t < 5; t++){
                    Y[i][j] += W[s][t] * X[i + s][j + t] + b;
                }
            }
        }
    }
}

void convolution2 (
            const float W[5][5], //(m, n)
            const float b, //(M - m + 1, N - n + 1)
            const float X[12][12], //(M, N)
            float Y[8][8] //(M - m + 1, N - n + 1)
            ) {

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            for (int s = 0; s < 5; s++) {
                for (int t = 0; t < 5; t++){
                    Y[i][j] += W[s][t] * X[i + s][j + t] + b;
                }
            }
        }
    }
}

//maxpooling層（順伝播）
void maxpooling1 (
                 const float X[24][24], //(M, N)
                 float Y[12][12] //(M - m + 1, N - n + 1)
                 ) {
    int n = 2;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            float temp = 0;
            for (int s = 0; s < 2; s++) {
                for (int t = 0; t < 2; t++){
                    if (X[n * i + s][n * j + t] > temp) {
                        temp = X[n * i + s][n * j + t];
                    }
                }
            }
            Y[i][j] = temp;
        }
    }

}

void maxpooling2 (
                 const float X[8][8], //(M, N)
                 float Y[4][4] //(M - m + 1, N - n + 1)
                 ) {
    int n = 2;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float temp = 0;
            for (int s = 0; s < 2; s++) {
                for (int t = 0; t < 2; t++){
                    if (X[n * i + s][n * j + t] > temp) {
                        temp = X[n * i + s][n * j + t];
                    }
                }
            }
            Y[i][j] = temp;
        }
    }

}

//softmax層（逆伝播）
void softmaxwithloss_bwd(int n, const float y[10], unsigned char t, float dEdx[10]) {
    
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

void relumatrix_bwd1(const float X[8][8], const float dY[8][8], float dX[8][8]) {
    for (int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            if (X[i][j] > 0) {
                dX[i][j] = dY[i][j];
            } else {
                dX[i][j] = 0;
            }
        }
    }
}

void relumatrix_bwd2(const float X[24][24], const float dY[24][24], float dX[24][24]) {
    for (int i = 0; i < 24; i++) {
        for(int j = 0; j < 24; j++) {
            if (X[i][j] > 0) {
                dX[i][j] = dY[i][j];
            } else {
                dX[i][j] = 0;
            }
        }
    }
}

//fc層（逆伝播）
void fc_bwd(
            const float x[16],    // (n,)
            const float dEdy[10], // (m,)
            const float A[160],    // (m, n)
            float dEdA[160],       // (m, n)
            float dEdb[10],       // (m,)
            float dEdx[16]        // (n,)
            ) {
    //dEdAの計算
    
    for (int i = 0; i < 10; i++){
        
        for (int j = 0; j < 16; j++){
            dEdA[j + i * 16] = dEdy[i] * x[j];
        }
    }
    //dEdbの計算
    
    for (int i = 0; i < 10; i++) {
        dEdb[i] = dEdy[i];
    }
    //下流へ転送する勾配
    
    for (int i = 0; i < 16; i++){
        dEdx[i] = 0;
        
        for (int j = 0; j < 10 ; j++){
            dEdx[i] += A[j * 16 + i] * dEdy[j];
        }
    }
}

//convolution層（逆伝播）
void convolution_bwd1(
                    float dX[][12],
                    float dY[][8],
                    float dW[][5], //(m, n)
                    float db,
                    float X[][12],
                    const float W[][5] //(m, n)
                    ){
                        for (int s = 0; s < 5; s++) {
                            for (int t = 0; t < 5; t++) {
                                for (int i = 0; i < 8; i++) {
                                    for (int j = 0; j < 8; j++) {
                                        dW[s][t] += dY[i][j] * X[i + s][j + t];
                                    }
                                }
                            }
                        }
                        for (int i = 0; i < 8; i++) {
                            for (int j = 0; j < 8; j++) {
                                db += dY[i][j];
                            }
                        }
                        for (int i = 0; i < 8; i++) {
                            for (int j = 0; j < 8; j++) {
                                for (int s = 0; s < 5; s++) {
                                    for (int t = 0; t < 5; t++){
                                        if (i - s < 0 || j - t < 0) {
                                            dY[i - s][j - t] = 0;
                                        }
                                        X[i][j] += dY[i - s][j - t] * W[s][t];
                                    }
                                }
                            }
                        }
}

void convolution_bwd2(
                    float dX[][24],
                    float dY[][28],
                    float dW[][5], //(m, n)
                    float db,
                    float X[][28],
                    const float W[][5] //(m, n)
                    ){
                        for (int s = 0; s < 5; s++) {
                            for (int t = 0; t < 5; t++) {
                                for (int i = 0; i < 24; i++) {
                                    for (int j = 0; j < 24; j++) {
                                        dW[s][t] += dY[i][j] * X[i + s][j + t];
                                    }
                                }
                            }
                        }
                        for (int i = 0; i < 24; i++) {
                            for (int j = 0; j < 24; j++) {
                                db += dY[i][j];
                            }
                        }
                        for (int i = 0; i < 24; i++) {
                            for (int j = 0; j < 24; j++) {
                                for (int s = 0; s < 5; s++) {
                                    for (int t = 0; t < 5; t++){
                                        if (i - s < 0 || j - t < 0) {
                                            dY[i - s][j - t] = 0;
                                        }
                                        X[i][j] += dY[i - s][j - t] * W[s][t];
                                    }
                                }
                            }
                        }
}

//maxpooling層（逆伝播）
void maxpooling_bwd1 (
                     float X[][8],
                     float Y[][4],
                     float dX[][8],
                     float dY[][4]
                     ) {
    int n = 2;
    for (int i = 0; i < 8 / n; i++) {
        for (int j = 0; j < 8 / n; j++) {
            for (int s = 0; s < n; s++) {
                for (int t = 0; t < n; t++){
                        if (Y[i][j] == X[i * n + s][j * n + t]){
                            dX[i * n + s][j * n + t] = dY[i][j];
                        } else {
                            dX[i * n + s][j * n + t] = 0;
                        }
                    }
                }
            }
        }
}

void maxpooling_bwd2 (
                     float X[][24],
                     float Y[][12],
                     float dX[][24],
                     float dY[][12]
                     ) {
    int n = 2;
    for (int i = 0; i < 24 / n; i++) {
        for (int j = 0; j < 24 / n; j++) {
            for (int s = 0; s < n; s++) {
                for (int t = 0; t < n; t++){
                        if (Y[i][j] == X[i * n + s][j * n + t]){
                            dX[i * n + s][j * n + t] = dY[i][j];
                        } else {
                            dX[i * n + s][j * n + t] = 0;
                        }
                    }
                }
            }
        }
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
    return - log(y[t] + 1e-7);
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

void save_vector(const char *filename, int n, const float *V){
    
    FILE *fp;
    if((fp = fopen(filename,"wb"))==NULL){
        printf("\aファイルをオープンできません。\n");
    } else {
        fwrite(V, sizeof(float), n, fp);

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



//SGD
void SGD(int m, 
         int n, 
         float dA[m*n], //(m, n)
         float dAave[m*n], //(m, n)
         float db[n], //(n,)
         float dbave[n], //(n,)
         float batch_f, 
         float learning_rate,  
         float A[m*n], //(m, n)
         float b[n] //(n,)
         ) {
    add(m * n, dA, dAave);
    add(n, db, dbave);
    scale(m * n, 1.0 / batch_f, dAave);
    scale(n, 1.0 / batch_f, dbave);
    scale(m * n, -1.0 * learning_rate, dAave);
    scale(n, -1.0 * learning_rate, dbave);
    add(m * n, dAave, A);
    add(n, dbave, b);
}

void SGDmatrix(int m, 
               int n, 
               float dA[][n], //(m, n)
               float dAave[][n], //(m, n)
               float db, //(n,)
               float dbave, //(n,)
               float batch_f, 
               float learning_rate,  
               float A[][n], //(m, n)
               float b //(n,))
               ) {
    addmatrix(m , n, dA, dAave);
    dbave += db;
    scalematrix(m , n, 1.0 / batch_f, dAave);
    dbave *= 1.0 / batch_f;
    scalematrix(m , n, -1.0 * learning_rate, dAave);
    dbave *= -1.0 * learning_rate;
    addmatrix(m , n, dAave, A);
    b += dbave;
               }


//推論（6層）
int inference6(const float W1[][5],
               const float W3[][5],
               const float A7[16 * 10], //(16,10)
               const float x[784],
               const float b1,
               const float b3,
               const float b7[10],
               float y8[10] //(10,)
               ){
    float X1[28][28];
    initmatrix(28,28,0,X1);
    makematrixc(28,28,X1,x);
    float Y1[24][24];
    float Y2[24][24];
    float Y3[12][12];
    float Y4[8][8];
    float Y5[8][8];
    float Y6[4][4];
    float y6[16];
    float y7[10];
    initmatrix(24,24,0,Y1);
    initmatrix(24,24,0,Y2);
    initmatrix(12,12,0,Y3);
    initmatrix(8,8,0,Y4);
    initmatrix(8,8,0,Y5);
    initmatrix(4,4,0,Y6);
    init(16,0,y6);
    init(10,0,y7);
    convolution1(W1,b1,X1,Y1);
    relumatrix1(Y1,Y2);
    maxpooling1(Y2,Y3);
    convolution2(W3,b3,Y3,Y4);
    relumatrix2(Y4,Y5);
    maxpooling2(Y5,Y6);
    unroll(4,4,Y6,y6);
    fc(4,4,y6,A7,b7,y7);
    softmax(10,y7,y8);
    int temp = 1;
    float M = 0;
    for (int i = 0; i < 10; i++){
        if (M < y6[i]){
            M = y8[i];
        }
    }
    for (int i = 0; i < 10; i++){
        if (M == y8[i])
        temp = i;
    }     

    return temp;

}

//back prop（6層）
void backward6(const float W1[5][5],
               const float W3[5][5],
               const float A7[16 * 10], //(16,10)
               const float x[784],
               float y8[10], //(10,)
               const float b1,
               const float b3,
               const float b7[10],
               unsigned char t,
               float dW1[5][5],
               float dW3[5][5],
               float db1,
               float db3,
               float dA7[16 * 10], //(16,10)
               float db7[10] //(4,)
               ){
    float X1[28][28];
    initmatrix(28,28,0,X1);
    makematrixc(28,28,X1,x);
    float Y1[24][24];
    float Y2[24][24];
    float Y3[12][12];
    float Y4[8][8];
    float Y5[8][8];
    float Y6[4][4];
    float y6[16];
    float y7[10];
    initmatrix(24,24,0,Y1);
    initmatrix(24,24,0,Y2);
    initmatrix(12,12,0,Y3);
    initmatrix(8,8,0,Y4);
    initmatrix(8,8,0,Y5);
    initmatrix(4,4,0,Y6);
    init(16,0,y6);
    init(10,0,y7);
    convolution1(W1,b1,X1,Y1);
    relumatrix1(Y1,Y2);
    maxpooling1(Y2,Y3);
    convolution2(W3,b3,Y3,Y4);
    relumatrix2(Y4,Y5);
    maxpooling2(Y5,Y6);
    unroll(4,4,Y6,y6);
    fc(4,4,y6,A7,b7,y7);
    softmax(10,y7,y8);
    float dX1[28][28];
    float dX2[24][24];
    float dX3[24][24];
    float dX4[12][12];
    float dX5[8][8];
    float dX6[8][8];
    float dX7[4][4];
    float dx7[16];
    float dx8[10];
    initmatrix(28,28,0,dX1);                 
    initmatrix(24,24,0,dX2);               
    initmatrix(24,24,0,dX3);
    initmatrix(12,12,0,dX4);
    initmatrix(12,12,0,dX5);
    initmatrix(8,8,0,dX6);
    initmatrix(4,4,0,dX7);
    init(16,0,dx7);
    init(10,0,dx8);
    softmaxwithloss_bwd(10,y8,t,dx8);
    fc_bwd(y6,dx8,A7,dA7,db7,dx7);
    makematrix(4,4,dX7,dx7);
    maxpooling_bwd1(Y5,Y6,dX6,dX7);
    relumatrix_bwd1(Y4,dX6,dX5);
    convolution_bwd1(dX4,dX5,dW3,db3,Y3,W3);
    maxpooling_bwd2(Y2,Y3,dX3,dX4);
    relumatrix_bwd2(Y1,dX3,dX2);
    convolution_bwd2(dX2,dX1,dW1,db1,X1,W1);

} 


// テスト
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
    if(argc != 4){
        printf("error");
        exit(1);
    }

    //ハイパーパラメータの設定
    int num_dim = atoi(argv[1]);
    int batch_size = atoi(argv[2]);
    int num_epoch = atoi(argv[3]);
    float learning_rate = 0;
    float batch_f = batch_size;
    int i, j, k, l;
    float rho1, rho2, eps;

    //変数の設定
    float W1[5][5];
    float W3[5][5];
    float A7[16 * 10];
    float x[784] = {0};
    float y8[10] = {0};
    float b1;
    float b3;
    float b7[10] = {0};
    unsigned char t;
    float dW1[5][5] = {0};
    float dW3[5][5] = {0};
    float db1 = {0};
    float db3 = {0};
    float dA7[16 * 10] = {0};
    float db7[10] = {0};  
    float dW1ave[5][5] = {0};
    float dW3ave[5][5] = {0};
    float db1ave = {0};
    float db3ave = {0};
    float dA7ave[16 * 10] = {0};
    float db7ave[10] = {0};  

    //変数メモリの確保
    int *index = malloc(sizeof(int) * train_count);
    float *acc_save_train = malloc(sizeof(float) *num_epoch);
    float *loss_save_train = malloc(sizeof(float) *num_epoch);
    float *acc_save_test = malloc(sizeof(float) *num_epoch);
    float *loss_save_test = malloc(sizeof(float) *num_epoch);


    //パラメタの初期化
    srand((unsigned)time(NULL));
    rand_initmatrix(5,5,W1);
    rand_initmatrix(5,5,W3);
    rand_init(16 * 10,A7);
    rand_init(4,b7);
    db1 = GetRandom(-1,1);
    db3 = GetRandom(-1,1);
    

    //ハイパーパラメータの確認と設定、optimizerの選択
    printf("batch : %d\n",batch_size);
    printf("dim : %d\n",num_dim);
    printf("epoch : %d\n",num_epoch);

    printf("your opitimizer is SGD\n");
    printf("Please input your learning rate : ");
    scanf("%f", &learning_rate);


    //[0 : N-1]配列の作成
    for (i = 0; i < train_count; i++){
        index[i] = i;
    }


    int num_train = train_count / batch_size;

    //確率的勾配降下法（エポック回数）
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        #pragma omp for
        for (i = 0; i < num_epoch; i++) {

            printf("======epoch %d / %d is running======\n\n", i + 1, num_epoch);
            //ランダムシャッフル
            shuffle(train_count, index);
            //勾配降下法（N/n回）
            #pragma omp for
            for (j = 0; j < num_train; j++) {

                //初期化 


                //学習
                #pragma omp for
                for (k = 0; k < batch_size; k++) { 
                    //back prop
                    printf("\r[%3d/100%%]", ((k + batch_size * j + 1) * 100) / train_count);
                    backward6(W1,W3,A7,train_x + 784 * index[100 * j + k], y8, b1, b3, b7, train_y[index[100 * j + k]], dW1, dW3, db1, db3, dA7, db7);

                    addmatrix(5 , 5, dW1, dW1ave);
                    db1ave += db1;
                    scalematrix(5 , 5, 1.0 / batch_f, dW1ave);
                    db1ave *= 1.0 / batch_f;
                    scalematrix(5 , 5, -1.0 * learning_rate, dW1ave);
                    db1ave *= -1.0 * learning_rate;
                    addmatrix(5 , 5, dW1ave, W1);
                    b1 += db1ave;

                    addmatrix(5 , 5, dW3, dW3ave);
                    db3ave += db3;
                    scalematrix(5 , 5, 1.0 / batch_f, dW3ave);
                    db3ave *= 1.0 / batch_f;
                    scalematrix(5 , 5, -1.0 * learning_rate, dW3ave);
                    db3ave *= -1.0 * learning_rate;
                    addmatrix(5 , 5, dW3ave, W3);
                    b3 += db3ave;

                        add(16 * 10, dA7, dA7ave);
                    add(10, db7, db7ave);
                    scale(16 * 10, 1.0 / batch_f, dA7ave);
                    scale(10, 1.0 / batch_f, db7ave);
                    scale(16 * 10, -1.0 * learning_rate, dA7ave);
                    scale(10, -1.0 * learning_rate, db7ave);
                    add(16 * 10, dA7ave, A7);
                    add(10, db7ave, b7);
                    SGD(16,10,dA7,dA7ave,db7,db7ave,batch_f,learning_rate,A7,b7);
                }
            
            }

            //正解率の確認
            int sum_train = 0;
            
            float loss_train = 0;
            float acc_train = 0;
            int sum_test = 0;
            float acc_test = 0;
            float loss_test = 0;
            #pragma omp for
            for (k = 0; k < train_count; k++) {
                if (inference6(W1,W3,A7, train_x + 784 * k, b1, b3, b7, y8) == train_y[k]) {
                    sum_train++;
                }
                loss_train += cross_entropy_error(y8, train_y[k]);
            }
            acc_train = sum_train * 100.0 / train_count;
            
            #pragma omp for
            for (k = 0; k < test_count; k++) {
                if (inference6(W1,W3,A7, train_x + 784 * k, b1, b3, b7, y8) == test_y[k]) {
                    sum_test++;
                }
                loss_test += cross_entropy_error(y8, test_y[k]);
            }
            acc_test = sum_test * 100.0 / test_count;
            printf("\n\naccuracy(train) : %f%%\n", acc_train);
            printf("loss(train) : %f\n\n", loss_train);
            printf("\naccuracy(test) : %f%%\n", acc_test);
            printf("loss(test) : %f\n\n", loss_test);
            printf("======completed======\n\n", i + 1);
            //各エポックごとの損失と正答率の格納
            loss_save_train[i] = loss_train;
            acc_save_train[i] = acc_train;
            loss_save_test[i] = loss_test;
            acc_save_test[i] = acc_test;
            }

        }
    
    //学習したパラメタの保存

    return 0;
}
