#include "math_tool.h"

#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>

using namespace std;
using namespace torch;

Tensor vec2SO3(Tensor vec) {
    return tensor({{0.f, -vec[2].item<float>(), vec[1].item<float>()},
                   {vec[2].item<float>(), 0.f, -vec[0].item<float>()},
                   {-vec[1].item<float>(), vec[0].item<float>(), 0.f}});
}

Tensor adj(Tensor T) {
    cout << T << endl;
    Tensor R = T.index({indexing::Slice(0, 3), indexing::Slice(0, 3)});

    Tensor v = T.index({indexing::Slice(0, 3), 3});

    return cat({cat({R, torch::zeros({3, 3})}, 1),
                cat({matmul(vec2SO3(v), R), R}, 1)});
}
