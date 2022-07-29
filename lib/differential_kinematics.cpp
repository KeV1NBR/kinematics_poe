#include "kinematics.h"

using namespace torch;
using namespace std;

Tensor Kinematics::jvp(Tensor theta, Tensor v, DeviceType device) {
    theta.requires_grad_(true);
    Tensor pos = this->forward(theta, device).requires_grad_(true);
    pos.backward(v);
    return theta.grad();
}
