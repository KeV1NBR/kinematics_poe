#include <ATen/Context.h>
#include <torch/cuda.h>

#include <ctime>
#include <iostream>

#include "torch/torch.h"

using namespace std;

int main() {
    std::cout << torch::cuda::is_available() << endl;
    for (int i = 0; i < 10; i++) {
        clock_t now = clock();
        torch::Tensor tensor = torch::rand({2, 3}).cuda();

        cout << tensor << endl;
        cout << tensor.t() << endl;
        tensor = tensor.matmul(tensor.t());
        cout << tensor.matmul(tensor.inverse()) << endl;

        cout << tensor << endl;
        clock_t end = clock();
        //    std::cout << tensor << std::endl;
        cout << double(end - now) / CLOCKS_PER_SEC << endl;
    }
    cout << endl;
    for (int i = 0; i < 10; i++) {
        clock_t now = clock();
        torch::Tensor tensor = torch::rand({2, 3});
        tensor.matmul(tensor.t());
        clock_t end = clock();
        //    std::cout << tensor << std::endl;
        cout << double(end - now) / CLOCKS_PER_SEC << endl;
    }
    return 0;
}
