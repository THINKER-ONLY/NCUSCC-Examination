import numpy as np
import time

def matrix_multiply(A, B):
    return np.dot(A, B)

def main():
    # 定义矩阵大小
    n = 10000
    # 创建两个大矩阵
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start_time = time.perf_counter()
    C = matrix_multiply(A, B)
    end_time = time.perf_counter()

    # 计算总执行时间
    total_execution_time = end_time - start_time
    print(f"Total execution time for 10000x10000 matrix multiplication: {total_execution_time} seconds")

if __name__ == '__main__':
    main()