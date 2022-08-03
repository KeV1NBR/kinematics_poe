#include <ATen/TensorIndexing.h>

#include "math_tool.h"

using namespace std;
using namespace torch;

Tensor vec2SE3(Tensor vec) {
    TensorOptions option =
        TensorOptions().dtype(kFloat).device(DeviceType::CPU);

    Tensor SO3 = vec2SO3(vec.index({indexing::Slice(indexing::None, 3)}));
    Tensor v = vec.index({indexing::Slice(3, indexing::None)});

    return torch::cat(
        {torch::cat({SO3, v.unsqueeze(1)}, 1), torch::zeros({1, 4}, option)},
        0);
}

Tensor SE32vec(torch::Tensor SE3) {
    TensorOptions option =
        TensorOptions().dtype(kFloat).device(DeviceType::CPU);

    return tensor(
        {SE3.index({2, 1}).item<float>(), SE3.index({0, 2}).item<float>(),
         SE3.index({1, 0}).item<float>(), SE3.index({0, 3}).item<float>(),
         SE3.index({1, 3}).item<float>(), SE3.index({2, 3}).item<float>()},
        option);
}

Tensor SE32se3(Tensor SE3) {
    TensorOptions option =
        TensorOptions().dtype(kFloat).device(DeviceType::CPU);

    Tensor omega =
        SO32vec(SE3.index({indexing::Slice(0, 3), indexing::Slice(0, 3)}));

    if (isNearZero(omega.norm().item<float>())) {
        Tensor R = torch::eye(3, option);
        Tensor v = SE3.index({indexing::Slice(0, 3), 3});

        return torch::cat({torch::cat({R, v.unsqueeze(1)}, 1),
                           torch::tensor({{0, 0, 0, 1}}, option)},
                          0);
    } else {
        Tensor theta = omega.norm();
        Tensor omegaMat =
            SE3.index({indexing::Slice(0, 3), indexing::Slice(0, 3)}) / theta;

        Tensor R =
            SO32so3(SE3.index({indexing::Slice(0, 3), indexing::Slice(0, 3)}));

        Tensor v =
            matmul(torch::eye(3, option) * theta + (1 - cos(theta)) * omegaMat +
                       (theta - sin(theta)) * matmul(omegaMat, omegaMat),
                   SE3.index({indexing::Slice(0, 3), 3}) / theta);

        return torch::cat({torch::cat({R, v.unsqueeze(1)}, 1),
                           torch::tensor({{0, 0, 0, 1}}, option)},
                          0);
    }
}
