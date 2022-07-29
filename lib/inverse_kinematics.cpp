#include <c10/core/DeviceType.h>

#include "kinematics.h"

using namespace std;
using namespace torch;

Tensor Kinematics::inverse(Tensor pos, Tensor initial) {
    return this->inverse(pos, initial, device);
}
Tensor Kinematics::inverse(Tensor pos, Tensor initial, DeviceType device) {
    Tensor Tsb = forward(initial, device);
    Tensor theta = initial.clone();

    Tensor Vs = matmul(adj(Tsb), );

    for (int i = 0; i < maxIteration; i++) {
        theta = theta + jvp(theta, Vs, device);
        Tsb = forward(theta, device);
    }

    return theta;
}
