#include <iostream>

__global__ void helloKernel() {
    printf("Hello from GPU! Thread %d\n", threadIdx.x);
}

int main() {
    helloKernel<<<1, 8>>>();
    cudaDeviceSynchronize();

    std::cout << "Hello from CPU!" << std::endl;

    return 0;
}