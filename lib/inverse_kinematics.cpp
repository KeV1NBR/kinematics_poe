#include <ATen/TensorIndexing.h>
#include <c10/core/DeviceType.h>

#include "kinematics.h"
#include "math_tool.h"

using namespace std;
using namespace torch;

Tensor Kinematics::inverse(Tensor pos, Tensor initial) {
    return this->inverse(pos, initial, device);
}
Tensor Kinematics::inverse(Tensor pos, Tensor initial, DeviceType device) {
    Tensor Tsb = forward(initial, device);
    Tensor theta = initial.clone();
    cout << "********" << endl;
    cout << adj(Tsb) << endl;
    cout << se32vec(SE32se3(matmul(SE3Inverse(Tsb), pos))) << endl;

    cout << "********" << endl;
    Tensor Vs =
        matmul(adj(Tsb), se32vec(SE32se3(matmul(SE3Inverse(Tsb), pos))));

    bool isErr = torch::norm(Vs.index({indexing::Slice(indexing::None, 3)}))
                         .item<float>() > this->eV ||
                 torch::norm(Vs.index({indexing::Slice(3, indexing::None)}))
                         .item<float>() > this->eOmega;
    for (int i = 0; i < 1; i++) {
        cout << "iter " << i << ":" << endl << Vs << endl;
        if (isErr == false) {
            return theta;
        }
        // theta = theta + jvp(theta, vec2se3(Vs), device);
        Tsb = forward(theta, device);

        Tensor Vs =
            matmul(adj(Tsb), se32vec(SE32se3(matmul(SE3Inverse(Tsb), pos))));

        isErr = torch::norm(Vs.index({indexing::Slice(indexing::None, 3)}))
                        .item<float>() > this->eV ||
                torch::norm(Vs.index({indexing::Slice(3, indexing::None)}))
                        .item<float>() > this->eOmega;
    }

    return theta;
}
