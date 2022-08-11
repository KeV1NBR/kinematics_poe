#include "kinematics.h"

using namespace torch;
using namespace std;

Tensor Kinematics::jvp(Tensor theta, Tensor v, DeviceType device) {
    Tensor thetaTmp = theta.clone();
    thetaTmp.requires_grad_(true);

    Tensor pos = this->forward(thetaTmp, device).requires_grad_(true);

    pos.backward(v);

    // cout << thetaTmp.grad() << endl;
    return thetaTmp.grad();
}
