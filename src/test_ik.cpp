#include <c10/core/DeviceType.h>

#include <utility>

#include "kinematics.h"
#include "math_tool.h"
#include "torch/torch.h"

using namespace std;
using namespace torch;

int main() {
    Kinematics km;
    torch::Tensor initial =
        torch::tensor({0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, torch::DeviceType::CPU);
    torch::Tensor theta =
        torch::tensor({0.f, 0.f, 0.f, 0.f, 0.f, 0.01f}, torch::DeviceType::CPU);

    cout << "theta: " << endl << theta << endl;
    torch::Tensor pos = km.forward(theta);
    //    cout << "pos:" << endl << pos << endl;
    Tensor result = km.inverse(pos, initial);
    // cout << "result:" << endl << result << endl;
    // cout << "pos:" << endl << pos << endl;
    // cout << "result pos:" << endl << km.forward(result) << endl;

    return 0;
}
