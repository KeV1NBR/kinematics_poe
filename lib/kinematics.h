#include <ATen/core/TensorBody.h>
#include <c10/core/DeviceType.h>

#include "torch/torch.h"

class Kinematics {
   public:
    Kinematics();
    ~Kinematics();
    std::vector<float> forward(std::vector<float> jointPosition);
    std::vector<float> forward(std::vector<float> jointPosition,
                               torch::DeviceType device);
    torch::Tensor forward(torch::Tensor theta);
    torch::Tensor forward(torch::Tensor theta, torch::DeviceType device);

    torch::Tensor inverse(torch::Tensor pos, torch::Tensor initial);
    torch::Tensor inverse(torch::Tensor pos, torch::Tensor initial,
                          torch::DeviceType device);

    void setDevice(torch::DeviceType device);

    torch::Tensor jvp(torch::Tensor theta, torch::Tensor v,
                      torch::DeviceType device);

   private:
    torch::DeviceType device;

    torch::Tensor M_cpu;
    torch::Tensor M_gpu;
    torch::Tensor S_cpu;
    torch::Tensor S_gpu;

    torch::Tensor* M;
    torch::Tensor* S;

    torch::Tensor SCalculate(torch::Tensor SList);

    float eOmega;
    float eV;
    int maxIteration;
};
