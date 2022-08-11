#include <ATen/TensorIndexing.h>

#include "math_tool.h"

using namespace std;
using namespace torch;

Tensor vec2se3(Tensor vec) {
    TensorOptions option =
        TensorOptions().dtype(kFloat).device(DeviceType::CPU);

    Tensor so3 = vec2so3(vec.index({indexing::Slice(indexing::None, 3)}));
    Tensor v = vec.index({indexing::Slice(3, indexing::None)});

    return torch::cat(
        {torch::cat({so3, v.unsqueeze(1)}, 1), torch::zeros({1, 4}, option)},
        0);
}

Tensor se32vec(torch::Tensor se3) {
    TensorOptions option =
        TensorOptions().dtype(kFloat).device(DeviceType::CPU);

    return tensor(
        {se3.index({2, 1}).item<float>(), se3.index({0, 2}).item<float>(),
         se3.index({1, 0}).item<float>(), se3.index({0, 3}).item<float>(),
         se3.index({1, 3}).item<float>(), se3.index({2, 3}).item<float>()},
        option);
}

Tensor se32SE3(Tensor se3) {
    TensorOptions option =
        TensorOptions().dtype(kFloat).device(DeviceType::CPU);

    Tensor omega =
        so32vec(se3.index({indexing::Slice(0, 3), indexing::Slice(0, 3)}));

    if (isNearZero(omega.norm().item<float>())) {
        Tensor R = torch::eye(3, option);
        Tensor v = se3.index({indexing::Slice(0, 3), 3});

        return torch::cat({torch::cat({R, v.unsqueeze(1)}, 1),
                           torch::tensor({{0, 0, 0, 1}}, option)},
                          0);
    } else {
        Tensor theta = omega.norm();
        Tensor omegaMat =
            se3.index({indexing::Slice(0, 3), indexing::Slice(0, 3)}) / theta;

        Tensor R =
            so32SO3(se3.index({indexing::Slice(0, 3), indexing::Slice(0, 3)}));

        Tensor v =
            matmul(torch::eye(3, option) * theta + (1 - cos(theta)) * omegaMat +
                       (theta - sin(theta)) * matmul(omegaMat, omegaMat),
                   se3.index({indexing::Slice(0, 3), 3}) / theta);

        return torch::cat({torch::cat({R, v.unsqueeze(1)}, 1),
                           torch::tensor({{0, 0, 0, 1}}, option)},
                          0);
    }
}

Tensor SE32se3(Tensor SE3) {
    TensorOptions option =
        TensorOptions().dtype(kFloat).device(DeviceType::CPU);

    Tensor R = SE3.index({indexing::Slice(0, 3), indexing::Slice(0, 3)});
    Tensor p = SE3.index({indexing::Slice(0, 3), 3});

    Tensor omegaMat = SO32so3(R);

    if (omegaMat.equal(torch::zeros({3, 3}, option))) {
        return torch::cat({torch::cat({omegaMat, p.unsqueeze(1)}, 1),
                           torch::tensor({{0, 0, 0, 0}}, option)},
                          0);
    } else {
        float theta = acos((trace(R) - 1.f) / 2.f).item<float>();
        Tensor v = matmul(torch::eye(3, option) - omegaMat / 2.f +
                              (1.f / theta - 1.f / tan(theta / 2.f) / 2.f) *
                                  matmul(omegaMat, omegaMat) / theta,
                          p);
        return torch::cat({torch::cat({omegaMat, v.unsqueeze(1)}, 1),
                           torch::tensor({{0, 0, 0, 0}}, option)},
                          0);
    }
}

Tensor SE3Inverse(Tensor SE3) {
    TensorOptions option =
        TensorOptions().dtype(kFloat).device(DeviceType::CPU);

    Tensor Rt = SE3.index({indexing::Slice(0, 3), indexing::Slice(0, 3)}).t();
    Tensor p = SE3.index({indexing::Slice(0, 3), 3});

    Tensor Rtp = -1 * matmul(Rt, p);

    return torch::cat({torch::cat({Rt, Rtp.unsqueeze(1)}, 1),
                       torch::tensor({{0, 0, 0, 1}}, option)},
                      0);
}
