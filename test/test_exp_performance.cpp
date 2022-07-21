#include <ATen/Context.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/linalg_matrix_exp.h>
#include <torch/cuda.h>

#include <ctime>
#include <iostream>

#include "torch/torch.h"

using namespace std;

int main() {
    std::cout << torch::cuda::is_available() << endl;

    for (int i = 0; i < 10; i++) {
        torch::Tensor tensor = torch::rand({2, 2}).cuda();
        tensor.exp();
    }

    clock_t now = clock();
    torch::Tensor tensor =
        torch::tensor({{0., M_PI / 3.}, {-M_PI / 3., 0.}}).cuda();
    for (int i = 0; i < 100; i++) {
        tensor.matrix_exp();
    }
    clock_t end = clock();
    //    std::cout << tensor << std::endl;
    cout << double(end - now) / CLOCKS_PER_SEC << endl;

    return 0;
}
