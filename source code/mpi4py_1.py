from mpi4py import MPI
import numpy as np
import time

def matrix_multiply(A, B):
    """执行矩阵乘法。"""
    return np.dot(A, B)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = 10000
    assert n % size == 0, "Matrix size must be divisible by the number of processes."

    block_size = n // size

    if rank == 0:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        C = np.empty((n, n), dtype=np.float64)
    else:
        A_local = np.empty((block_size, n), dtype=np.float64)
        B_local = np.empty((n, block_size), dtype=np.float64)
        C_local = np.empty((block_size, block_size), dtype=np.float64)

    requests = []
    if rank == 0:
        for i in range(1, size):
            A_slice = A[i*block_size:(i+1)*block_size, :].copy()  # Ensure contiguous
            B_slice = B[:, i*block_size:(i+1)*block_size].copy()  # Ensure contiguous
            req1 = comm.Isend(A_slice, dest=i, tag=11)
            req2 = comm.Isend(B_slice, dest=i, tag=12)
            requests.extend([req1, req2])
        A_local = A[0:block_size, :]
        B_local = B[:, 0:block_size]
    else:
        comm.Recv(A_local, source=0, tag=11)
        comm.Recv(B_local, source=0, tag=12)

    # 计算局部矩阵乘法
    start_time = time.perf_counter()
    C_local = matrix_multiply(A_local, B_local)
    end_time = time.perf_counter()

    # 等待非阻塞通信完成
    if rank == 0:
        for req in requests:
            req.Wait()

    # 收集所有进程的局部结果和执行时间
    execution_time = end_time - start_time
    if rank == 0:
        total_time = np.empty(size, dtype=np.float64)
        C[0:block_size, 0:block_size] = C_local
        for i in range(1, size):
            temp_C_local = np.empty((block_size, block_size), dtype=np.float64)
            comm.Recv(temp_C_local, source=i, tag=11)
            temp_time = np.empty(1, dtype=np.float64)
            comm.Recv(temp_time, source=i, tag=11)
            C[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = temp_C_local
            total_time[i] = temp_time[0]
        average_time = np.mean(total_time)
        print(f"Average execution time: {average_time} seconds")
    else:
        comm.Send(C_local, dest=0, tag=11)
        comm.Send(np.array([execution_time], dtype=np.float64), dest=0, tag=11)
if __name__ == '__main__':
    main()