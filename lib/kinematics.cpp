#include "kinematics.h"

#include <c10/core/DeviceType.h>

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
Tensor TM_S() {
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

    return torch::stack(
        {
            (cat({omega1, v1}, 0)),
            (cat({omega2, v2}, 0)),
            (cat({omega3, v3}, 0)),
            (cat({omega4, v4}, 0)),
            (cat({omega5, v5}, 0)),
            (cat({omega6, v6}, 0)),
        },
        1);
}

Kinematics::Kinematics() : device(DeviceType::CPU) {
    M_cpu = TM_M();
    M_gpu = M_cpu.cuda();

    S_cpu = TM_S();
    S_gpu = S_cpu.cuda();

    this->M = &M_cpu;
    this->S = &S_cpu;
}
Kinematics::~Kinematics() {}

void Kinematics::setDevice(DeviceType device) { this->device = device; }
vector<float> Kinematics::forward(const vector<float>& jointPosition) {
    return this->forward(jointPosition, this->device);
}
vector<float> Kinematics::forward(const vector<float>& jointPosition,
                                  DeviceType device) {
    this->M = (device == DeviceType::CPU) ? &M_cpu : &M_gpu;
    this->S = (device == DeviceType::CPU) ? &S_cpu : &S_gpu;
}
