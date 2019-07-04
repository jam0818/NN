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
	return min + (float)(rand()*(max-min)/(RAND_MAX));
}

//初期化関数
void rand_init(int n, float * o) {
    for(int i = 0; i < n; i++) {
        o[i] = GetRandom(-1, 1);
    }
}

//fc層（順伝播）
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
    #pragma omp parallel for
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
    #pragma omp parallel for
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
            const float *x,    // (n,)
            const float *dEdy, // (m,)
            const float *A,    // (m, n)
            float *dEdA,       // (m, n)
            float *dEdb,       // (m,)
            float *dEdx        // (n,)
            ) {
    //dEdAの計算
    #pragma omp parallel for
    for (int i = 0; i < m; i++){
        #pragma omp parallel for
        for (int j = 0; j < n; j++){
            dEdA[j + i * n] = dEdy[i] * x[j];
        }
    }
    //dEdbの計算
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        dEdb[i] = dEdy[i];
    }
    //下流へ転送する勾配
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        dEdx[i] = 0;
        #pragma omp parallel for
        for (int j = 0; j < m ; j++){
            dEdx[i] += A[j * n + i] * dEdy[j];
        }
    }
}


//ランダムシャッフル
void shuffle(int n, int *x){
    srand(time(NULL));
    #pragma omp parallel for
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

//推論（6層）
int inference6(const float*A1,const float *b1,const float*A3,const float*b3,const float*A5,const float*b5, const float *x){
    float *y1 = malloc(sizeof(float) * 50); // (50,)
    float *y2 = malloc(sizeof(float) * 50); // (50,)
    float *y3 = malloc(sizeof(float) * 100); // (100,)
    float *y4 = malloc(sizeof(float) * 100); // (100,)
    float *y5 = malloc(sizeof(float) * 10); // (50,)
    float *y6 = malloc(sizeof(float) * 10); // (50,)
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
    free(y6);
    return temp;

}

//back prop（6層）
void backward6(const float *A1, // (50, 784)
               const float *b1, // (50,)
               const float *A3, // (100, 50)
               const float *b3, // (50,)
               const float *A5, // (10, 50)
               const float *b5, // (10,)
               const float *x, // (784,) 
               unsigned char t, // (1,)
               float *y6, //(10,)
               float *dA1, // (50, 784)
               float *db1, // (50,)
               float *dA3, // (100, 50)
               float *db3, // (50,)
               float *dA5, // (10, 50)
               float *db5 // (10,)
               ){
    float *y1 = malloc(sizeof(float) * 50); // (50,)
    float *y2 = malloc(sizeof(float) * 50); // (50,)
    float *y3 = malloc(sizeof(float) * 100); // (100,)
    float *y4 = malloc(sizeof(float) * 100); // (100,)
    float *y5 = malloc(sizeof(float) * 10); // (10,)

    //順伝播は右端の引数が出力値
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y2);
    fc(100, 50, y2, A3, b3, y3);
    relu(100, y3, y4);
    fc(10, 100, y4, A5, b5, y5);
    softmax(10, y5, y6);

    float *dx6 = malloc(sizeof(float) * 10);
    float *dx5 = malloc(sizeof(float) * 100);
    float *dx4 = malloc(sizeof(float) * 100);
    float *dx3 = malloc(sizeof(float) * 50);
    float *dx2 = malloc(sizeof(float) * 50);
    float *dx1 = malloc(sizeof(float) * 784);

    softmaxwithloss_bwd(10, y6, t, dx6);
    fc_bwd(10, 100, y4, dx6, A5, dA5, db5, dx5);
    relu_bwd(100, y3, dx5, dx4);
    fc_bwd(100, 50, y2, dx4, A3, dA3, db3, dx3);
    relu_bwd(50, y1, dx3, dx2);
    fc_bwd(50, 784, x, dx2, A1, dA1, db1, dx1);
    free(y1);
    free(y2);
    free(y3);
    free(y4);
    free(y5);
    free(dx1);
    free(dx2);
    free(dx3);
    free(dx4);
    free(dx5);
    free(dx6);
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
    float learning_late = 0;
    float batch_f = batch_size;
    int i, j, k, l;


    //変数メモリの確保
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
    float *dA5ave = malloc(sizeof(float) * 100 * 10);
    float *dA3ave = malloc(sizeof(float) * 50 * 100);
    float *dA1ave = malloc(sizeof(float) * 784 * 50);
    float *db5ave = malloc(sizeof(float) * 10);
    float *db3ave = malloc(sizeof(float) * 100);
    float *db1ave = malloc(sizeof(float) * 50);
    int *index = malloc(sizeof(int) * train_count);
    float * acc = malloc(sizeof(float) * num_epoch);
    float * loss = malloc(sizeof(float) * num_epoch);

    //パラメタの初期化
    srand((unsigned)time(NULL));
    rand_init(784 * 50, A1);
    rand_init(50, b1);
    rand_init(50 * 100, A3);
    rand_init(100, b3);
    rand_init(100 * 10, A5);
    rand_init(10, b5);

    //ハイパーパラメータの確認と設定
    printf("batch : %d\n",batch_size);
    printf("dim : %d\n",num_dim);
    printf("epoch : %d\n",num_epoch);
    printf("Please input your learning rate : ");
    scanf("%f", &learning_late);
    printf("learning rate : %.2f\n", learning_late);



    //[0 : N-1]配列の作成
    for (i = 0; i < train_count; i++){
        index[i] = i;
    }


    int num_train = train_count / batch_size;
    float train_f = num_train;

    //確率的勾配降下法（エポック回数）
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        #pragma omp for
        for (i = 0; i < num_epoch; i++) {
            printf("epoch %d / %d is running...\n\n", i + 1, num_epoch);

            //ランダムシャッフル
            shuffle(train_count, index);
            //勾配降下法（N/n回）
            #pragma omp for
            for (j = 0; j < num_train; j++) {
                //初期化 
                init(784 * 50, 0, dA1ave);
                init(50 * 100, 0, dA3ave);
                init(100 * 10, 0, dA5ave);
                init(50, 0, db1ave);
                init(100, 0, db3ave);
                init(10, 0, db5ave);

                //学習
                #pragma omp for
                for (k = 0; k < batch_size; k++) {
                
                    //back prop
                    backward6(A1, b1, A3, b3, A5, b5, train_x + 784 * index[100 * j + k], train_y[index[100 * j + k]], y6, dA1, db1, dA3, db3, dA5, db5);
                
                    //aveの計算
                    add(784 * 50, dA1, dA1ave);
                    add(50 * 100, dA3, dA3ave);
                    add(100 * 10, dA5, dA5ave);
                    add(50, db1, db1ave);
                    add(100, db3, db3ave);
                    add(10, db5, db5ave);
                    scale(784 * 50, 1.0 / batch_f, dA1ave);
                    scale(50 * 100, 1.0 / batch_f, dA3ave);
                    scale(100 * 10, 1.0 / batch_f, dA5ave);
                    scale(50, 1.0 / batch_f, db1ave);
                    scale(100, 1.0 / batch_f, db3ave);
                    scale(10, 1.0 / batch_f, db5ave);
                    scale(784 * 50, -1.0 * learning_late, dA1ave);
                    scale(50 * 100, -1.0 * learning_late, dA3ave);
                    scale(100 * 10, -1.0 * learning_late, dA5ave);
                    scale(50, -1.0 * learning_late, db1ave);
                    scale(100, -1.0 * learning_late, db3ave);
                    scale(10, -1.0 * learning_late, db5ave);

                    //パラメタの更新
                    add(784 * 50, dA1ave, A1);
                    add(50 * 100, dA3ave, A3);
                    add(100 * 10, dA5ave, A5);
                    add(50, db1ave, b1);
                    add(100, db3ave, b3);
                    add(10, db5ave, b5);

                
                }
            
                //プログレスバー
                if (j == 0){
                    printf("0%%      100%%\n");
                    printf("+--------+\n", i + 1);
                }
                if (j % (num_train / 10) == 0) {
                    printf("#");
                }
            }

            //正解率の確認
            int sum_train = 0;
            float loss_train = 0;
            float acc_train = 0;
            #pragma omp for
            for (k = 0; k < test_count; k++) {
                if (inference6(A1, b1, A3, b3, A5, b5, test_x + 784 * k) == test_y[k]) {
                sum_train++;
                }
            }
            acc_train = sum_train * 100.0 / test_count;
            printf("\naccuracy : %f%%\n\n", acc_train);
            printf("completed...\n\n", i + 1);
            #pragma omp for
            for (l = 0; l < 10; l++) {
                loss_train += cross_entropy_error(y6, l);
            }

            //各エポックごとの損失と正答率の格納
            loss[i] = loss_train;
            acc[i] = acc_train;
        }
    }
    //学習したパラメタの保存
    save("param1.dat", 50, 784, A1, b1);
    save("param3.dat", 100, 50, A3, b3);
    save("param5.dat", 10, 100, A5, b5);
    save_vector("acc_train.dat", num_epoch, acc);
    return 0;
}
