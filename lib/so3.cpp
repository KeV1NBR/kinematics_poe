#include <ATen/TensorIndexing.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/matrix_exp.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "math_tool.h"

using namespace std;
using namespace torch;

bool isNearZero(float val) { return fabs(val) < 1e-6f; }

Tensor vec2SO3(Tensor vec) {
    TensorOptions option =
        TensorOptions().dtype(kFloat).device(DeviceType::CPU);

    return tensor({{0.f, -vec[2].item<float>(), vec[1].item<float>()},
                   {vec[2].item<float>(), 0.f, -vec[0].item<float>()},
                   {-vec[1].item<float>(), vec[0].item<float>(), 0.f}},
                  option);
}

Tensor SO32vec(torch::Tensor SO3) {
    TensorOptions option =
        TensorOptions().dtype(kFloat).device(DeviceType::CPU);

    return tensor({SO3.index({2, 1}).item<float>(),  //
                   SO3.index({0, 2}).item<float>(),  //
                   SO3.index({1, 0}).item<float>()},
                  option);
}

Tensor SO32so3(Tensor SO3) {
    TensorOptions option =
        TensorOptions().dtype(kFloat).device(DeviceType::CPU);

    //// torch implememtation
    // return matrix_exp(SO3);

    // formula implementation
    Tensor omega = SO32vec(SO3);
    Tensor theta = omega.norm();
    if (isNearZero(theta.item<float>())) {
        return torch::eye(3, option);
    }

    Tensor I = torch::eye(3, option);

    Tensor omegaMat = SO3 / theta;

    return I + torch::sin(theta) * omegaMat +
           (1 - torch::cos(theta)) * matmul(omegaMat, omegaMat);
}

Tensor so32SO3(Tensor so3) {
    TensorOptions option =
        TensorOptions().dtype(kFloat).device(DeviceType::CPU);

    // formula implementation p.85 3.58~3.61

    // trR = 1 + 2cos(θ)
    float cosTheta = (torch::trace(so3).item<float>() - 1) / 2;

    if (cosTheta >= 1) {
        return torch::zeros({3, 3}, option);
    } else if (cosTheta <= -1) {
        if (isNearZero(1.f + so3.index({2, 2}).item<float>())) {
            Tensor omega =
                (1.0 / sqrt(2 * (1 + so3.index({2, 2}).item<float>()))) *
                tensor({so3.index({0, 2}).item<float>(),
                        so3.index({1, 2}).item<float>(),
                        1.f + so3.index({2, 2}).item<float>()},
                       option);
            return vec2SO3(omega * M_PI);
        } else if (isNearZero(1.f + so3.index({1, 1}).item<float>())) {
            Tensor omega =
                (1.0 / sqrt(2 * (1 + so3.index({1, 1}).item<float>()))) *
                tensor({so3.index({0, 1}).item<float>(),
                        1.f + so3.index({1, 1}).item<float>(),
                        so3.index({2, 1}).item<float>()},
                       option);
            return vec2SO3(omega * M_PI);
        } else {
            Tensor omega =
                (1.0 / sqrt(2 * (1 + so3.index({0, 0}).item<float>()))) *
                tensor({1.f + so3.index({0, 0}).item<float>(),
                        so3.index({1, 0}).item<float>(),
                        so3.index({2, 0}).item<float>()},
                       option);
            return vec2SO3(omega * M_PI);
        }
    } else {
        float theta = acos(cosTheta);
        return (theta / 2.f / sin(theta)) * (so3 - so3.t());
    }
}
Tensor adj(Tensor T) {
    cout << T << endl;
    Tensor R = T.index({indexing::Slice(0, 3), indexing::Slice(0, 3)});

    Tensor v = T.index({indexing::Slice(0, 3), 3});

    return cat({cat({R, torch::zeros({3, 3})}, 1),
                cat({matmul(vec2SO3(v), R), R}, 1)});
}
