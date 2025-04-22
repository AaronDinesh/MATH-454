#include <stdio.h>

void mul(float a, float b){
    printf("%f", a*b);   
}

void mul(int a, int b){
    printf("%d", a*b);
}


int main(){
    mul(2., 3.);
    mul((int) 2, (int)3);


}