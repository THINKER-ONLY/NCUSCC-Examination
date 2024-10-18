import numpy as np
from joblib import Parallel, delayed
import time

# 定义矩阵乘法函数
def matrix_multiply(A, B):
    return np.dot(A, B)

# 定义执行单个区块乘法的函数
def block_multiply(args):
    A_block, B_block = args
    return matrix_multiply(A_block, B_block)

def main():
    n = 10000
    num_blocks = 4

    block_size = n // num_blocks

    num_runs = 5
    total_time = 0

    for _ in range(num_runs):
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)

        # 初始化最终矩阵C
        C = np.zeros((n, n), dtype=np.float64)

        start_time = time.time()

        # 分割矩阵并执行多进程计算
        block_args = [(A[i*block_size:(i+1)*block_size, :], B[:, i*block_size:(i+1)*block_size]) for i in range(num_blocks)]
        block_results = Parallel(n_jobs=-1)(delayed(block_multiply)(args) for args in block_args)

        # 将区块结果合并到最终矩阵
        for i, block_result in enumerate(block_results):
            row_start = i // num_blocks * block_size
            col_start = (i % num_blocks) * block_size
            C[row_start:row_start+block_size, col_start:col_start+block_size] = block_result

        end_time = time.time()
        total_time += end_time - start_time

    # 计算平均时间
    average_time = total_time / num_runs

    # 仅在程序结束时输出平均时间
    print(f"Average execution time: {average_time:.4f} seconds")

if __name__ == "__main__":
    main()