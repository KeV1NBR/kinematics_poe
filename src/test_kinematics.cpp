#include <c10/core/DeviceType.h>

#include "kinematics.h"
#include "math_tool.h"
#include "torch/torch.h"

using namespace std;
using namespace torch;

int main() {
    Kinematics km;
    torch::Tensor v =
        torch::tensor({1.57079632f, 0.f, 0.f, 0.f, 2.35619449f, 2.35619449f},
                      torch::DeviceType::CPU);

    cout << v << endl;
    cout << vec2se3(v) << endl;
    cout << se32SE3(vec2se3(v)) << endl;
    cout << SE32se3(se32SE3(vec2se3(v))) << endl;

    cout << SE3Inverse(se32SE3(vec2se3(v))) << endl;

    cout << matmul(se32SE3(vec2se3(v)), SE3Inverse(se32SE3(vec2se3(v))));

    // cout << km.jvp(theta, torch::DeviceType::CPU);
    // vector<float> pos = {M_PI_2, M_PI_2, M_PI_2, M_PI_2, M_PI_2, M_PI_2};

    // for (int i = 0; i < 10; i++) km.forward(pos,
    // torch::DeviceType::CUDA);

    // clock_t now = clock();
    // for (int i = 0; i < 10; i++) km.forward(pos,
    // torch::DeviceType::CUDA); clock_t end = clock(); cout << double(end -
    // now) / CLOCKS_PER_SEC << endl;

    return 0;
}
