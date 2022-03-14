#include "rotation.h"
#include "bmpfuncs.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
	//arg[0]=프로그램네임
	//arg[1]=읽어올 파일명
	//arg[2]=저장할 파일명
	//arg[3]=회전시킬 각도
    if (argc < 4) {
        printf("Usage: %s <src file> <dst file> <degree>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int width, height;
    float* input = readImage(argv[1], &width, &height);
    float* output = (float*)malloc(sizeof(float) * width * height);

    rotate(input, output, width, height, argv[3]);
    storeImage(output, argv[2], height, width, argv[1]);

    return 0;
}