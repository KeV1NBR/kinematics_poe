#include "kinematics.h"

#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include <iostream>

using namespace std;
using namespace torch;

Tensor TM_M() {
    float l1 = .425f;
    float l2 = .3922f;
    float w1 = .1333f;
    float w2 = .0996f;
    float h1 = .1625f;
    float h2 = .0997f;

    return tensor({
        {-1.f, 0.f, 0.f, l1 + l2},
        {0.f, 0.f, 1.f, w1 + w2},
        {0.f, 1.f, 0.f, h1 - h2},
        {0.f, 0.f, 0.f, 1.f},
    });
}
Tensor TM_SList() {
    float l1 = .425f;
    float l2 = .3922f;
    float w1 = .1333f;
    float w2 = .0996f;
    float h1 = .1625f;
    float h2 = .0997f;

    Tensor omega1 = tensor({0.f, 0.f, 1.f});
    Tensor omega2 = tensor({0.f, 1.f, 0.f});
    Tensor omega3 = tensor({0.f, 1.f, 0.f});
    Tensor omega4 = tensor({0.f, 1.f, 0.f});
    Tensor omega5 = tensor({0.f, 0.f, -1.f});
    Tensor omega6 = tensor({0.f, 1.f, 0.f});

    Tensor q1 = tensor({0.f, 0.f, h1});
    Tensor q2 = tensor({0.f, 0.f, h1});
    Tensor q3 = tensor({l1, 0.f, h1});
    Tensor q4 = tensor({l1 + l2, 0.f, h1});
    Tensor q5 = tensor({l1 + l2, w1, h1});
    Tensor q6 = tensor({l1 + l2, w1, h1 - h2});

    Tensor v1 = -1 * cross(omega1, q1);
    Tensor v2 = -1 * cross(omega2, q2);
    Tensor v3 = -1 * cross(omega3, q3);
    Tensor v4 = -1 * cross(omega4, q4);
    Tensor v5 = -1 * cross(omega5, q5);
    Tensor v6 = -1 * cross(omega6, q6);

    return torch::stack({(cat({omega1, v1}, 0)), (cat({omega2, v2}, 0)),
                         (cat({omega3, v3}, 0)), (cat({omega4, v4}, 0)),
                         (cat({omega5, v5}, 0)), (cat({omega6, v6}, 0))},
                        1);
}

Kinematics::Kinematics() : device(DeviceType::CPU) {
    M_cpu = TM_M();
    //    M_gpu = M_cpu.cuda();

    Tensor SList = TM_SList();

    S_cpu = SCalculate(TM_SList());
    //    S_gpu = S_cpu.cuda();

    this->M = &M_cpu;
    this->S = &S_cpu;
}
Kinematics::~Kinematics() {}

void Kinematics::setDevice(DeviceType device) { this->device = device; }
vector<float> Kinematics::forward(vector<float> jointPosition) {
    return this->forward(jointPosition, this->device);
}

vector<float> Kinematics::forward(vector<float> jointPosition,
                                  DeviceType device) {
    this->M = (device == DeviceType::CPU) ? &M_cpu : &M_gpu;
    this->S = (device == DeviceType::CPU) ? &S_cpu : &S_gpu;

    Tensor res = torch::eye(4, device);
    Tensor theta =
        torch::from_blob(jointPosition.data(), {int(jointPosition.size())},
                         torch::TensorOptions().dtype(torch::kFloat));
    Tensor STheta = S->clone().detach();
    for (int i = 0; i < 6; i++) {
        res = matmul(res, (STheta[i] * theta[i]).matrix_exp());
    }
    res = matmul(res, *M);
    // cout << res;
    return vector<float>();
}
Tensor Kinematics::SCalculate(Tensor SList) {
    Tensor res;

    for (int i = 0; i < SList.sizes()[1]; i++) {
        Tensor SVector = SList.index({indexing::Slice(), i});

        Tensor omega = tensor(
            {{0.f, -SVector[2].item<float>(), SVector[1].item<float>()},
             {SVector[2].item<float>(), 0.f, -SVector[0].item<float>()},
             {-SVector[1].item<float>(), SVector[0].item<float>(), 0.f}});
        Tensor v = SVector.index({indexing::Slice(3, indexing::None)});

        Tensor tmp = cat({omega, v.unsqueeze(1)}, 1);
        tmp = cat({tmp, torch::zeros({1, 4})}, 0);

        if (i == 0) {
            res = tmp.unsqueeze(0);
            continue;
        }
        res = cat({res, tmp.unsqueeze(0)}, 0);
    }

    return res;
}
