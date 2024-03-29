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
void add(int n, const float * x, float *o) {
    for(int i = 0; i < n; i++){
        o[i] = x[i] + o[i];
    }
}

//ベクトルの足し算
void addscale(int n, float s,const float * x, float *o) {
    for(int i = 0; i < n; i++){
        o[i] = s * x[i] + o[i];
    }
}


//指定した値で初期化
void init(int n, float x, float *o) {
    for(int i = 0; i < n; i++) {
        o[i] = x;
    }
}

//行列の表示
void print(int m, int n, const float *x) {
    for(int i = 0; i < m ; i++){
        for(int j = 0; j < n; j++){
            printf("%.4f ", x[j + i * n]);
        }
        printf("\n");
    }
}

//スカラー倍
void scale(int n, float x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] *= x;
    }    
}


//一定の範囲での乱数生成関数
float GetRandom(float min,float max){
	return min + (float)(rand()*(max-min)/(RAND_MAX));
}

//初期化関数
void rand_init(int n, float *o) {
    for(int i = 0; i < n; i++) {
        o[i] = GetRandom(-1, 1);
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
        const float *x, // (n,)
        const float *A, // (m, n)
        const float *b, // (m,)
        float *y // (m,)
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
void relu(int n, const float *x, float *y) {
    
    for (int i = 0; i < n; i++) {
        if (x[i] < 0) {
            y[i] = 0;
        } else {
            y[i] = x[i];
        }
    }
}


//softmax層（順伝播）
void softmax(int n, const float *x, float *y) {
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
void convolution(int m,
            int n,
            int M,
            int N,
            const float *W, //(m, n)
            const float *b,
            const float *X, //(M, N)
            float *Y //(M - m + 1, N - n + 1)
            ) {
    float bs = *b;
    //print(M,N,X);
    //printf("\n");
    //printf("%f\n\n",bs);
    //print(5,5,W);
    //printf("\n");
    for (int i = 0; i < M - m + 1; i++) {
        for (int j = 0; j < N - n + 1; j++) {
            for (int s = 0; s < m; s++) {
                for (int t = 0; t < n; t++){
                    Y[j + i * (N - n + 1)] += W[t + s * n] * X[t + s * N + i * N + j] + bs;
                }
            }
        }
    }
    //print(M-m+1,N-n+1,Y);
    //printf("\n");
}

//maxpooling層（順伝播）
void maxpooling (int n,
                 int M,
                 int N,
                 const float *X, //(M, N)
                 float *Y //(M / n, N / n)
                 ) {
    for (int i = 0; i < M / n; i++) {
        for (int j = 0; j < N / n; j++) {
            float temp = 0;
            for (int s = 0; s < n; s++) {
                for (int t = 0; t < n; t++){
                    if (X[j * n + i * n * N + t + s * N] > temp) {
                        temp = X[j * n + i * n * N + t + s * N];
                    }
                }
            }
            Y[j + i * (N / n)] = temp;
        }
    }

}

//softmax層（逆伝播）
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx) {
    
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
void fc_bwd(int m,
            int n,
            const float *x, // (n,)
            const float *dEdy, // (m,)
            const float *A, // (m, n)
            float *dEdA, // (m, n)
            float *dEdb,  // (m,)
            float *dEdx // (n,)
            ) {
    //dEdAの計算
    
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            dEdA[j + i * n] = dEdy[i] * x[j];
        }
    }
    //dEdbの計算
    
    for (int i = 0; i < m; i++) {
        dEdb[i] = dEdy[i];
    }
    //下流へ転送する勾配
    
    for (int i = 0; i < n; i++){
        dEdx[i] = 0;
        for (int j = 0; j < m ; j++){
            dEdx[i] += A[j * n + i] * dEdy[j];
        }
    }
}

//convolution層（逆伝播）
void convolution_bwd(int m,
                    int n,
                    int M,
                    int N,
                    float *dY, //(M - m + 1, N - n + 1)
                    float *dW, //(m, n)
                    float *db,
                    const float *X, //(M, N)
                    const float *W, //(m, n)
                    float *dX //(M, N)
                    ){
    //print(M - m + 1, N - n + 1, dY);
    //printf("dY\n");
    float dbs = *db;
    for (int s = 0; s < m; s++)
    {
        for (int t = 0; t < n; t++)
        {
            for (int i = 0; i < M - m + 1; i++)
            {
                for (int j = 0; j < N - n + 1; j++)
                {
                    dW[s * n + t] += dY[i * (N - n + 1) + j] * X[t + s * N + i * N + j];
                }
            }
        }
                        }
                        for (int i = 0; i < M - m + 1; i++) {
                            for (int j = 0; j < N - n + 1; j++) {
                                dbs += dY[i * (N - n + 1) + j];
                            }
                        }

                        //パディング
                        float *dY_p = malloc(sizeof(float) * (N + n - 1)*(M + m - 1));
                        for (int i = 0; i < m - 1; i++) {
                            for (int j = 0; j < N + n - 1; j++) {
                                    dY_p[j + i*(N + n - 1)] = 0;
                            }
                        }
                        for (int i = m - 1; i < M; i++) {
                            for (int j = 0; j < N + n - 1; j++) {
                                if(j < n - 1 || j > N - 1) {
                                    dY_p[j + i*(N + n - 1)] = 0;
                                } else {
                                    dY_p[j + i*(N + n - 1)] = dY[j - (n - 1) + (i - (m - 1)) * (N - n + 1)];
                                }
                            }
                        }

                        for (int i = M; i < M + m - 1; i++)
                        {
                            for (int j = 0; j < N + n - 1; j++)
                                {
                                    dY_p[j + i*(N + n - 1)] = 0;
                                }
                        }
                        //print(M + m - 1, N + n - 1,dY_p);
                        //printf("dY_p\n");
                        //print(m, n, W);
                        //printf("W\n");
                        for (int i = 0; i < M; i++) {
                            for (int j = 0; j < N; j++) {
                                for (int s = 0; s < m; s++) {
                                    for (int t = 0; t < n; t++){
                                        dX[j + i * N] += dY_p[i*(N + n - 1) + j + t + s * (N + n - 1)] * W[(m - 1 - s) * n + (n - 1 - t)];
                                    }
                                }
                            }
                        }
                        //print(M, N, dX);
                        //printf("dX\n");
                        free(dY_p);
}

//maxpooling層（逆伝播）
void maxpooling_bwd (int n,
                     int M,
                     int N,
                     float *X, //(M, N)
                     float *Y, //(M / n, N / n)
                     float *dY, //(M / n, N / n)
                     float *dX //(M, N)) 
                    ){
    //print((M / n),(N / n),Y);
    //printf("Y\n");
    //print(M,N,X);
    //printf("X\n");
    for (int i = 0; i < M / n; i++) {
        for (int j = 0; j < N / n; j++) {
            for (int s = 0; s < n; s++) {
                for (int t = 0; t < n; t++){
                        if (Y[j + i * (N / n)] == X[j * n + i * n * N + t + s * N]){
                            dX[j * n + i * n * N + t + s * N] = dY[j + i * (N / n)];
                        } else {
                            dX[j * n + i * n * N + t + s * N] = 0;
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

void save_kernel(const char *filename, int m, int n, const float *A, const float*b){
    
    FILE *fp;
    if((fp = fopen(filename,"wb"))==NULL){
        printf("\aファイルをオープンできません。\n");
    } else {
        fwrite(A, sizeof(float), m * n, fp);
        fwrite(b, sizeof(float), 1, fp);
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






//推論（6層）
int inference6(const float *W1, //(5, 5)
               const float *W3, //(5, 5)
               const float *A7, //(16, 10)
               const float *b1, 
               const float *b3,
               const float *b7, //(10,)
               const float *x, //(28, 28)
               float *y8 //(10,)
               ){
    //print(5,5,W1);
    //printf("W1\n");
    //print(5,5,W3);
    //printf("W3\n");
    float *y1 = malloc(sizeof(float) *24*24);
    float *y2 = malloc(sizeof(float) *24*24);
    float *y3 = malloc(sizeof(float) *12*12);
    float *y4 = malloc(sizeof(float) *8*8);
    float *y5 = malloc(sizeof(float) *8*8);
    float *y6 = malloc(sizeof(float) *4*4);
    float *y7 = malloc(sizeof(float) *10);

    init(24*24,0,y1);
    init(24*24,0,y2);
    init(12*12,0,y3);
    init(8*8,0,y4);
    init(8*8,0,y5);
    init(4*4,0,y6);
    init(10,0,y7);
    convolution(5,5,28,28,W1,b1,x,y1);
    //print(24,24,y1);
    //printf("y1\n");
    relu(24*24,y1,y2);
    //print(24,24,y2);
    //printf("y2\n");
    maxpooling(2,24,24,y2,y3);
    //print(12,12,y3);
    //printf("y3\n");
    convolution(5,5,12,12,W3,b3,y3,y4);
    //print(8,8,y4);
    //printf("y4\n");
    relu(8*8,y4,y5);
    //rint(8,8,y5);
    //printf("y5\n"); 
    maxpooling(2,8,8,y5,y6);
    //print(4,4,y6);
    //printf("y6\n");
    //print(10, 16, A7);
    //printf("A7\n");
    //print(1, 10, b7);
    //printf("b7\n");
    fc(10,16,y6,A7,b7,y7);
    //print(1,10,y7);
    //printf("y7\n");
    softmax(10,y7,y8);
    //print(1,10,y8);
    //printf("y8\n");
    int temp = 1;
    float M = 0;
    for (int i = 0; i < 10; i++){
        if (M < y8[i]){
            M = y8[i];
        }
    }
    for (int i = 0; i < 10; i++){
        if (M == y8[i])
        temp = i;
    }
    free(y1);
    free(y2);
    free(y3);
    free(y4);
    free(y5);
    free(y6);
    free(y7);

    return temp;

}

//back prop（6層）
void backward6(const float *W1, //(5, 5)
               const float *W3, //(5, 5)
               const float *A7, //(16, 10)
               const float *x, //(28, 28)
               float *y8, //(10,)
               const float *b1, 
               const float *b3,
               const float *b7, 
               unsigned char t,
               float *dW1, //(5, 5)
               float *dW3, //(5, 5)
               float *db1,
               float *db3,
               float *dA7, //(16, 10)
               float *db7 
               ){
    //print(5,5,W1);
    //printf("W1\n");
    //print(5,5,W3);
    //printf("W3\n");
    float *y1 = malloc(sizeof(float) *24*24);
    float *y2 = malloc(sizeof(float) *24*24);
    float *y3 = malloc(sizeof(float) *12*12);
    float *y4 = malloc(sizeof(float) *8*8);
    float *y5 = malloc(sizeof(float) *8*8);
    float *y6 = malloc(sizeof(float) *4*4);
    float *y7 = malloc(sizeof(float) *10);

    init(24*24,0,y1);
    init(24*24,0,y2);
    init(12*12,0,y3);
    init(8*8,0,y4);
    init(8*8,0,y5);
    init(4*4,0,y6);
    init(10,0,y7);
    convolution(5,5,28,28,W1,b1,x,y1);
    //print(24,24,y1);
    //printf("y1\n");
    relu(24*24,y1,y2);
    //print(24,24,y2);
    //printf("y2\n");
    maxpooling(2,24,24,y2,y3);
    //print(12,12,y3);
    //printf("y3\n");
    convolution(5,5,12,12,W3,b3,y3,y4);
    //print(8,8,y4);
    //printf("y4\n");
    relu(8*8,y4,y5);
    //rint(8,8,y5);
    //printf("y5\n"); 
    maxpooling(2,8,8,y5,y6);
    //print(4,4,y6);
    //printf("y6\n");
    //print(10, 16, A7);
    //printf("A7\n");
    //print(1, 10, b7);
    //printf("b7\n");
    fc(10,16,y6,A7,b7,y7);
    //print(1,10,y7);
    //printf("y7\n");
    softmax(10,y7,y8);
    //print(1,10,y8);
    //printf("y8\n");
    float *dx1 = malloc(sizeof(float) *28*28);
    float *dx2 = malloc(sizeof(float) *24*24);
    float *dx3 = malloc(sizeof(float) *24*24);
    float *dx4 = malloc(sizeof(float) *12*12);
    float *dx5 = malloc(sizeof(float) *8*8);
    float *dx6 = malloc(sizeof(float) *8*8);
    float *dx7 = malloc(sizeof(float) *16);
    float *dx8 = malloc(sizeof(float) *10);
    init(28*28,0,dx1);
    init(24*24,0,dx2);
    init(24*24,0,dx3);
    init(12*12,0,dx4);
    init(8*8,0,dx5);
    init(8*8,0,dx6);
    init(16,0,dx7);
    init(10,0,dx8);
    softmaxwithloss_bwd(10,y8,t,dx8);
    //print(1,10,dx8);
    //printf("dx8\n");    
    fc_bwd(10,16,y6,dx8,A7,dA7,db7,dx7);
    //print(10, 16, dA7);
    //printf("dA7\n");
    //print(4,4,dx7);
    //printf("dx7\n"); 
    maxpooling_bwd(2,8,8,y5,y6,dx7,dx6);
    //print(8,8,dx6);
    //printf("dx6\n");  
    relu_bwd(8*8,y4,dx6,dx5);
    convolution_bwd(5,5,12,12,dx5,dW3,db3,y3,W3,dx4);
    //print(12, 12, dx4);
    //printf("dx4\n");  
    maxpooling_bwd(2,24,24,y2,y3,dx4,dx3);    
    relu_bwd(24*24,y1,dx3,dx2);
    convolution_bwd(5,5,28,28,dx2,dW1,db1,x,W1,dx1);
    //print(5,5,dW1);
    //printf("dW1\n");
    //print(5,5,dW3);
    //printf("dW3\n");
    free(y1);
    free(y2);
    free(y3);
    free(y4);
    free(y5);
    free(y6);
    free(y7);

    free(dx1);
    free(dx2);
    free(dx3);
    free(dx4);
    free(dx5);
    free(dx6);
    free(dx7);
    free(dx8);


} 
void SGD(int m, 
         int n, 
         float *dAave, //(m, n)
         float *dbave, //(n,)
         float batch_f, 
         float learning_rate,  
         float *A, //(m, n)
         float *b
         ) {
    scale(m * n, -1.0 * learning_rate, dAave);
    scale(n, -1.0 * learning_rate, dbave);
    add(m * n, dAave, A);
    add(n, dbave, b);
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


    //ハイパーパラメータの設定
    int num_dim = 784;//atoi(argv[1]);
    int batch_size = 100;//atoi(argv[2]);
    int num_epoch = 10;//atoi(argv[3]);
    float learning_rate = 0.2;
    float batch_f = batch_size;
    int i, j, k;

    //変数の設定


    //変数メモリの確保
    int *index = malloc(sizeof(int) * train_count);

    float *y8 = malloc(sizeof(float) *10);
    float *W1 = malloc(sizeof(float) * 5 * 5);
    float *W3 = malloc(sizeof(float) * 5 * 5);
    float *A7 = malloc(sizeof(float) * 10 * 16);
    float b1 = 0;
    float b3 = 0;
    float *b7 = malloc(sizeof(float) * 10);
    float *dW1ave = malloc(sizeof(float) * 5 * 5);
    float *dW3ave = malloc(sizeof(float) * 5 * 5);
    float *dA7ave = malloc(sizeof(float) * 10 * 16);
    float *db7ave = malloc(sizeof(float) * 10);
    float db1ave = 0;
    float db3ave = 0;
    float db1 = 0;
    float db3 = 0;
    float *dW1 = malloc(sizeof(float) * 5 * 5);
    float *dW3 = malloc(sizeof(float) * 5 * 5);
    float *dA7 = malloc(sizeof(float) * 10 * 16);
    float *db7 = malloc(sizeof(float) * 10);
    float * acc_save_train = malloc(sizeof(float) * num_epoch);
    float * loss_save_train = malloc(sizeof(float) * num_epoch);
    float * acc_save_test = malloc(sizeof(float) * num_epoch);
    float * loss_save_test = malloc(sizeof(float) * num_epoch);
    //パラメタの初期化
    srand((unsigned)time(NULL));
    he_init(5*5,W1);
    he_init(5*5,W3);
    he_init(10*16,A7);
    he_init(10,b7);
    b1 = GetRandom(0,1);
    b3 = GetRandom(0,1);
    //ハイパーパラメータの確認と設定、optimizerの選択
    printf("batch : %d\n",batch_size);
    printf("dim : %d\n",num_dim);
    printf("epoch : %d\n",num_epoch);

    printf("your opitimizer is SGD\n");
    printf("Please input your learning rate : ");


    //[0 : N-1]配列の作成
    for (i = 0; i < train_count; i++){
        index[i] = i;
    }


    int num_train = train_count / batch_size;

    //確率的勾配降下法（エポック回数）

    {
        
        for (i = 0; i < num_epoch; i++) {

            printf("======epoch %d / %d is running======\n\n", i + 1, num_epoch);
            //ランダムシャッフル
            shuffle(train_count, index);
            //勾配降下法（N/n回）
            
            for (j = 0; j < num_train; j++) {

                //初期化 
                init(5*5,0,dW1);
                init(5*5,0,dW3);
                init(5*5,0,dW1ave);
                init(5*5,0,dW3ave);
                init(10*16,0,dA7ave);
                init(10*16,0,dA7);
                init(10,0,db7);
                init(10,0,db7ave);
                db1ave = 0;
                db3ave = 0;
                db1 = 0;
                db3 = 0;

                //学習
                
                for (k = 0; k < batch_size; k++) { 
                    //back prop
                    printf("\r[%3d/100%%]", ((k + batch_size * j + 1) * 100) / train_count);
                    init(5*5,0,dW1);
                    init(5*5,0,dW3);
                    db1 = 0;
                    db3 = 0;

                    backward6(W1,W3,A7,train_x + 784 * index[batch_size * j + k],y8,&b1, &b3,b7,train_y[index[batch_size * j + k]],dW1,dW3,&db1,&db3,dA7,db7);
                   // print(10, 16, dA7);
                    //printf("dA7\n");
                    addscale(5 * 5,1/batch_f, dW1, dW1ave);
                    addscale(1, 1/batch_f,&db1, &db3ave);
                    addscale(5 * 5,1/batch_f, dW3, dW3ave);
                    addscale(1,1/batch_f, &db3, &db3ave);
                    addscale(10*16,1/batch_f, dA7, dA7ave);
                    addscale(1, 1/batch_f,db7, db7ave);       
                    SGD(5,5,dW1ave,&db1ave,batch_f,learning_rate,W1,&b1);
                    SGD(5,5,dW3ave,&db3ave,batch_f,learning_rate,W3,&b3);
                    SGD(16,10,dA7ave,db7ave,batch_f,learning_rate,A7,b7);   
                    //print(5, 5, dW1);
                    //printf("dW1\n");
                    //print(5, 5, dW3);
                    //printf("dW3\n");
                    //print(5, 5, dW1ave);
                    //printf("dW1\n");
                    //print(5, 5, dW3ave);
                    //printf("dW3\n");
                }
          
                //print(10, 16,dA7ave);
            }

            //正解率の確認
            //正解率の確認
            int sum_train = 0;
            float loss_train = 0;
            float acc_train = 0;
            int sum_test = 0;
            float acc_test = 0;
            float loss_test = 0;
            
            print(5,5,W1);
            printf("W1\n");
            print(5,5,W3);
            printf("W3\n");
            for (k = 0; k < train_count; k++) {
                if (inference6(W1,W3,A7,&b1,&b3,b7, train_x + 784 * k, y8) == train_y[k]) {
                    sum_train++;
                }
                loss_train += cross_entropy_error(y8, train_y[k]);
            }
            acc_train = sum_train * 100.0 / train_count;

            
            
            for (int l = 0; l < test_count;  l++)
            {
                if (inference6(W1,W3,A7,&b1,&b3,b7, test_x + 784 * l, y8) == test_y[l]) {
                    sum_test++;
                }
                loss_test += cross_entropy_error(y8, test_y[k]);
            }
            acc_test = sum_test * 100.0 / test_count;
            printf("\n\naccuracy(train) : %f%%\n", acc_train);
            printf("loss(train) : %f\n\n", loss_train / train_count);
            printf("\naccuracy(test) : %f%%\n", acc_test);
            printf("loss(test) : %f\n\n", loss_test / test_count);
            printf("======completed======\n\n", i + 1);
            //各エポックごとの損失と正答率の格納
            loss_save_train[i] = loss_train;
            acc_save_train[i] = acc_train;
            loss_save_test[i] = loss_test;
            acc_save_test[i] = acc_test;           


        }
    }
    //学習したパラメタの保存
    save_kernel("paramW1.dat", 5, 5, W1, &b1);
    save_kernel("paramW3.dat", 5, 5, W3, &b3);
    save("paramA7", 16, 10, A7, b7);
    return 0;
}
