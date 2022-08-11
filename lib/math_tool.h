#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

torch::Tensor vec2so3(torch::Tensor vec);
/**
 * formula: A = a^
 *
 * rotation vector to skew symmetric matrix
 * vector(3) φ {r,p,y} to so(3) Φ {{ 0, -y,  p},
 *                                 { y,  0, -r},
 *                                 {-p,  r,  0}}
 * **/
torch::Tensor vec2se3(torch::Tensor vec);

torch::Tensor so32vec(torch::Tensor so3);
/**
 * formula : a = Aˇ
 *
 * skew symmetric matrix to rotation vector
 *
 * so(3) Φ {{ 0, -y,  p}, to rotation vector(3) φ {r,p,y}
 *          { y,  0, -r},
 *          {-p,  r,  0}}
 * **/
torch::Tensor se32vec(torch::Tensor SE3);

torch::Tensor so32SO3(torch::Tensor so3);
/**
 * formula : R = exp(Φ)
 * **/
torch::Tensor se32SE3(torch::Tensor se3);
torch::Tensor SO32so3(torch::Tensor SO3);
/**
 * formula : Φ = log(R)
 * **/
torch::Tensor SE32se3(torch::Tensor SE3);

torch::Tensor SE3Inverse(torch::Tensor SE3);
torch::Tensor adj(torch::Tensor T);

bool isNearZero(float val);
