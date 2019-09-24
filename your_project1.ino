#include "ESC2nd.h"
#define DELTA_T 0.010
#define KP 0.000165
#define KI 0.00013
#define KD 0.0001
float param_L[4] = {12,5,3,1};
float param_R[4] = {1,3,5,12};
float t_val[8] = {3968 ,3368 ,2300 ,1920 ,1920 ,2300 ,3368 ,3968};
float diff_L[2];
float diff_R[2];
float integral;
float p,i,d;
float err;
Motor motorR('r');
Motor motorL('l');
Sensor sensor(DEVICE_ADDR1);

float PID_left(float val0, float val1, float val2,float val3,float t_val0,float t_val1,float t_val2,float t_val3) {
  float p, i, d;

  diff_L[0] = diff_L[1];
  diff_L[1] = param_L[0]*(val0 - t_val0) + param_L[1]*(val1 - t_val1) + param_L[2]*(val2 - t_val2) + param_L[3]*(val3 - t_val3);
  integral += (diff_L[1] + diff_L[0]) / 2.0 * DELTA_T;

  p = KP * diff_L[1];
  i = KI * integral;
  d = KD * (diff_L[1] + diff_L[0]) / DELTA_T;

  err = p + d + i;

  return err;
}

float PID_right(float val4, float val5, float val6,float val7,float t_val4,float t_val5,float t_val6,float t_val7) {
  float p, i, d;

  diff_R[0] = diff_R[1];
  diff_R[1] = param_R[0]*(val4 - t_val4) + param_R[1]*(val5 - t_val5) + param_R[2]*(val6 - t_val6) + param_R[3]*(val7 - t_val7);
  integral += (diff_R[1] + diff_R[0]) / 2.0 * DELTA_T;

  p = KP * diff_R[1];
  i = KI * integral;
  d = KD * (diff_R[1] + diff_R[0]) / DELTA_T;

  err = p  + d + i;

  return err;
}
void setup() {
  Serial.begin(9600);
  delay(1000);
}

void loop() {
  int val[8];
  float fval[8];
  sensor.read(val);
  for(int i = 0; i < 8; i++){
    fval[i] = val[i];
  }
  int pid_L_val = PID_left(fval[0],fval[1],fval[2],fval[3],t_val[0],t_val[1],t_val[2],t_val[3]);
  int pid_R_val = PID_right(fval[4],fval[5],fval[6],fval[7],t_val[4],t_val[5],t_val[6],t_val[7]);
  int Power_R = constrain(30 + pid_R_val,-100,200);
  int Power_L = constrain(30 + pid_L_val,-100,200);
  motorR.drive(Power_R);
  motorL.drive(Power_L);
}
