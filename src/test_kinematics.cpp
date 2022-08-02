#include <c10/core/DeviceType.h>

#include "kinematics.h"
#include "math_tool.h"
#include "torch/torch.h"

using namespace std;
using namespace torch;

int main() {
    Kinematics km;
    torch::Tensor v = torch::tensor({1.f, 2.f, 3.f}, torch::DeviceType::CPU);

    torch::Tensor SO3 = vec2SO3(v);
    torch::Tensor so3 = SO32so3(SO3);
    torch::Tensor SO3_2 = so32SO3(so3);
    torch::Tensor v_2 = SO32vec(SO3_2);

    cout << v << endl;
    cout << SO3 << endl;
    cout << so3 << endl;
    cout << SO3_2 << endl;
    cout << v_2 << endl;

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
