import numpy as np
from multiprocessing import Pool
import time

def matrix_multiply(A, B):
    return np.dot(A, B)

def split_matrix(A, B, num_splits):
    row_size = A.shape[0] // num_splits
    col_size = B.shape[1] // num_splits

    row_splits = [A[i*row_size:(i+1)*row_size] for i in range(num_splits)]
    col_splits = [B[:, j*col_size:(j+1)*col_size] for j in range(num_splits)]

    return row_splits, col_splits

def parallel_matrix_multiply(A, B, num_splits):
    row_size = A.shape[0] // num_splits
    col_size = B.shape[1] // num_splits
    final_result = np.empty((A.shape[0], B.shape[1]), dtype=np.float32)  # 预先定义空矩阵

    pool = Pool(processes=num_splits)
    row_splits, col_splits = split_matrix(A, B, num_splits)
    
    # 使用生成器来创建任务
    tasks = ((row_splits[i], col_splits[i]) for i in range(num_splits))
    
    results = pool.starmap(matrix_multiply, tasks)

    pool.close()
    pool.join()

    # 直接在最终结果矩阵上进行操作
    for i in range(num_splits):
        start_row = i * row_size
        end_row = (i + 1) * row_size
        start_col = i * col_size
        end_col = (i + 1) * col_size
        final_result[start_row:end_row, start_col:end_col] = results[i]

    return final_result

def main():
    n = 10000
    num_splits = 4
    num_runs = 5  # 定义运行的次数
    total_time_all_runs = 0  # 用于记录所有运行的总时间

    for _ in range(num_runs):
        A = np.random.rand(n, n).astype(np.float32)  # 使用 float32 数据类型
        B = np.random.rand(n, n).astype(np.float32)

        starttime = time.time()
        result = parallel_matrix_multiply(A, B, num_splits)
        total_time_all_runs += time.time() - starttime

    # 计算平均时间
    average_time = total_time_all_runs / num_runs
    print(f"Average execution time: {average_time} seconds")

if __name__ == "__main__":
    main()