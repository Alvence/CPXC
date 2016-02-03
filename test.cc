#include <stdio.h>

#include "CP.cc"

int main(){
  int c[2][2];
  c[0][0] = 5;
  c[0][1] = 5;
  c[1][0] = 25;
  c[1][1] = 65;
  printf("mi = %f",MI(c,100));
  expectedMI(c,100);
}
