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

    cout << vec2SE3(v) << endl;
    cout << SE32se3(vec2SE3(v)) << endl;
    cout << se32SE3(SE32se3(vec2SE3(v))) << endl;

    cout << se3Inverse(SE32se3(vec2SE3(v))) << endl;

    cout << matmul(SE32se3(vec2SE3(v)), se3Inverse(SE32se3(vec2SE3(v))));

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
