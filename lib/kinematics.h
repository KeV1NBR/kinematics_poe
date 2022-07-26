#include "torch/torch.h"

class Kinematics {
   public:
    Kinematics();
    ~Kinematics();
    std::vector<float> forward(std::vector<float> jointPosition);
    std::vector<float> forward(std::vector<float> jointPosition,
                               torch::DeviceType device);

    void setDevice(torch::DeviceType device);

   private:
    torch::DeviceType device;

    torch::Tensor M_cpu;
    torch::Tensor M_gpu;
    torch::Tensor S_cpu;
    torch::Tensor S_gpu;

    torch::Tensor* M;
    torch::Tensor* S;

    torch::Tensor SCalculate(torch::Tensor SList);
};
