import sys
sys.path.append('..')

from time import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import modern_robotics as mr
from tqdm import tqdm

from kinematics.kinematics import *
from robot_model import tm5_model

def benchmark_parallel_iterate(device, n_iterations=20):
    M, Slist = tm5_model.getModel()
    kinematics = Kinematics(M, Slist, device)
    times = []
    for i in range(n_iterations):
        thetalist = torch.rand(2**i, 5, device=kinematics.M.device)
        start = time()
        _ = kinematics.forward(thetalist)
        took = time() - start
        times.append(took)
        print("GPU Iteration {0} took {1}".format(i, took))
    return times


def benchmark_sequential_iterate(n_iterations=20):
    M, Slist = tm5_model.getModel()
    times = []
    for i in range(n_iterations):
        thetalist = torch.rand(2**i, 5)
        start = time()
        all_r = []
        for t in tqdm(thetalist):
            all_r.append(mr.FKinSpace(M, Slist, t.cpu().numpy()))
        _ = np.stack(all_r, axis=0)
        took = time() - start
        times.append(took)
        print("CPU Iteration {0} took {1}".format(i, took))
    return times


def benchmark_sequential(n_iterations=10):
    # Warm up
    _ = benchmark_sequential_iterate(5)
    return benchmark_sequential_iterate(n_iterations)


def benchmark_parallel(device, n_iterations=15):
    # Warm up
    _ = benchmark_parallel_iterate(device, 5)
    return benchmark_parallel_iterate(device, n_iterations)


def main():
    cpu_times = benchmark_sequential()
    np.save("cpu_times.npy", cpu_times)
    vectorized_times = benchmark_parallel("cpu")
    np.save("vectorized_times.npy", vectorized_times)
    gpu_times = benchmark_parallel("cuda:0")
    np.save("gpu_times.npy", gpu_times)

    plt.plot(cpu_times, label="Modern robotics")
    plt.plot(vectorized_times, label="Ours (CPU)")
    plt.plot(gpu_times, label="Ours (GPU) ")
    plt.legend()
    plt.xlabel("Number of input configurations (log scale)")
    plt.ylabel("Time [s]")
    plt.savefig("benchmark.eps")
    plt.show()


if __name__ == "__main__":
    main()
