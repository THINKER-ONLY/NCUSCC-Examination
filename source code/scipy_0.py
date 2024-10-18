import numpy as np
from scipy.linalg import blas
import time

def matrix_multiply(A, B):
    return blas.ddot(A, B)

def main():
    n = 10000  # 定义矩阵的大小
    num_runs = 5  # 定义运行的次数
    total_time_all_runs = 0  # 用于记录所有运行的总时间

    for _ in range(num_runs):
        A = np.random.rand(n, n).astype(np.float64)  # 使用 float64 数据类型
        B = np.random.rand(n, n).astype(np.float64)

        starttime = time.time()
        result = matrix_multiply(A, B)
        total_time_all_runs += time.time() - starttime

    # 计算平均时间
    average_time = total_time_all_runs / num_runs
    print(f"Average execution time: {average_time} seconds")

if __name__ == "__main__":
    main()