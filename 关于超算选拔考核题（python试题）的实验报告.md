#   **超算选拔考核题（python试题）的实验报告**

**作者：THINKER-ONLY**      **Tel：13803957693      QQ：3193304954     mail：3193304954@qq.com**

**实验要求：完成大型矩阵的乘法并不断优化，比较性能差异并绘制表图。**

## 一.实验环境的搭建过程。

### **1.配置虚拟机：**在虚拟机中安装 Ubuntu 22.04 LTS 操作系统。

​       我使用的虚拟机软件是VMware，其实更好的选择是使用WSL2，但因为VMware的快照功能给我更多的容错。所以我在思考后还是选择了使用VMware作为工具。在此之前我已经安装了基于Ubuntu操作系统的虚拟机，但由于本次实验使用的Ubuntu的版本为22.04LTS，这与我所使用的24.04LTS不同，所以我创建了新的虚拟机。

### 2.配置虚拟机的网络连接，确保可以正常联网。

​        对于Ubuntu虚拟机的网络链接，在正确安装后会自动配置完成，无需在此赘述。

### 3.安装并配置Python环境。

​        由于Ubuntu操作系统优秀的图形化界面可以直接在应用商店中安装`vscode`。

​        或者也可以到官网进行下载。官网下载地址：[https://code.visualstudio.com](https://code.visualstudio.com/),找到下载好的安装包，双击打开，一直点下一步即可。

​        首先进行中文的配置，这一步其实可有可无，如果你英语足够好，可以直接略过此步奏。

​        在扩展这一栏，我们搜索chinese，选择第一个进行instill即可。

<img src="https://www.helloimg.com/i/2024/10/19/67133dad1f7f6.png" alt="屏幕截图 2024-10-17 193609.png" title="屏幕截图 2024-10-17 193609.png" />

​        然后在扩展这一栏，我们搜索Python，选择第一个进行instill即可。

<img src="https://www.helloimg.com/i/2024/10/19/67133daac7046.png" alt="屏幕截图 2024-10-14 101918.png" title="屏幕截图 2024-10-14 101918.png" />

​        完成后新建一个文件夹来存放我们的代码，这里我创建的文件夹名称为vscode。
<img src="https://www.helloimg.com/i/2024/10/19/67133daa64629.png" alt="屏幕截图 2024-10-14 102309.png" title="屏幕截图 2024-10-14 102309.png" />
​        最后在vscode中打开我们创建的文件夹。
<img src="https://www.helloimg.com/i/2024/10/19/67133daa6858e.png" alt="屏幕截图 2024-10-14 102513.png" title="屏幕截图 2024-10-14 102513.png" />

​        再根据下图步骤进行必要的设置。
<img src="https://www.helloimg.com/i/2024/10/19/67133da70f3e3.png" alt="39acda2dab69e7aaecdf3a89069e4b38.png" title="39acda2dab69e7aaecdf3a89069e4b38.png" />
<img src="https://www.helloimg.com/i/2024/10/19/67133da6f2cda.png" alt="b97aed9ca30d25a07afa20f5eab87396.png" title="b97aed9ca30d25a07afa20f5eab87396.png" />

​       点击后会自动创建一个settings.json文件，在settings.json中输入下列代码，用来配置flake8和yapf并关闭pylint工具。

```
{
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "yapf",
    "python.linting.flake8Args": ["--max-line-length=248"],
    "python.linting.pylintEnabled": false
}
```

​       保存并关闭即可。最后可以创建一个新的.py后缀的文件进行输出一下"Hello World"看看是否配置成功。

​       然后我们需要对计算所需要的安装包进行安装和配置。另外在查阅教程的时候发现对于mpi4py的使用需要一些前置的步骤来确保其正常运行，使用如下代码来进行MPI的安装：

```Linux
sudo apt-get update
sudo apt-get install mpich libmpich-dev  # 安装 MPICH
# 或者
sudo apt-get update
sudo apt-get install openmpi-bin libopenmpi-dev  # 安装 OpenMPI
```

> [!WARNING]
>
> 如果两个都进行安装，可能会造成后续指令的冲突，导致程序无法正常运行。

​       这里我安装的是MPICH。

​       在正确安装后可以使用如下指令来确保你的mpy4pi库安装成功：

```Linux
python -c "from mpi4py import MPI"
```

​        而对于另外的指定库multiprocessing而言，我也遇到了第一个问题。对于这个问题我也在"实验过程中遇到的问题及解决方案"中给出了解决方案。

## 二.不同并行策略的性能对比及对大规模矩阵运算的加速效果。

​        平时我们的矩阵计算，无论计算机的性能如何，都只能在同一时间内处理一个计算，而对于该题目，其实质在于通过分割矩阵，再将矩阵分发给不同的进程，来同时并行处理，使得计算机能在较短的时间内完成，平时单线程无法完成的大量数据计算，实现缩短计算时间，提高计算机性能的利用率。为了方便演示，代码统一使用四进程来控制变量。

### 1.使用mpi4py 实现一个大规模矩阵的乘法。

​          对于mpi4py库，我们可以通过如下代码实现：

```python
from mpi4py import MPI
import numpy as np
import time

def matrix_multiply(A, B):
    """执行矩阵乘法。"""
    return np.dot(A, B)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  # 获取总进程数

    n = 10000
    # 确保矩阵可以被进程数整除
    assert n % size == 0, "Matrix size must be divisible by the number of processes."

    # 分块大小
    block_size = n // size

    if rank == 0:
        # 主进程初始化矩阵A和B
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        # 初始化用于收集结果的数组
        C = np.empty((n, n), dtype=np.float64)
    else:
        # 其他进程只需要分配空间给局部矩阵
        A_local = np.empty((block_size, n), dtype=np.float64)
        B_local = np.empty((n, block_size), dtype=np.float64)
        # 初始化用于收集结果的数组
        C_local = np.empty((block_size, block_size), dtype=np.float64)

    # 主进程分发矩阵A和B到所有进程
    if rank == 0:
        for i in range(1, size):
            # 确保发送的数组是连续的
            A_slice = A[i*block_size:(i+1)*block_size, :].copy()
            B_slice = B[:, i*block_size:(i+1)*block_size].copy()
            comm.Send(A_slice, dest=i, tag=11)
            comm.Send(B_slice, dest=i, tag=12)
        # 主进程保留第一块，避免重复计算
        A_local = A[0:block_size, :]
        B_local = B[:, 0:block_size]
        C_local = np.empty((block_size, block_size), dtype=np.float64)
    else:
        # 其他进程接收矩阵A和B的局部部分
        A_local = np.empty((block_size, n), dtype=np.float64)
        B_local = np.empty((n, block_size), dtype=np.float64)
        comm.Recv(A_local, source=0, tag=11)
        comm.Recv(B_local, source=0, tag=12)

    # 所有进程计算其分配到的矩阵片段的乘法
    start_time = time.perf_counter()
    C_local = matrix_multiply(A_local, B_local)
    end_time = time.perf_counter()

    # 计算执行时间
    execution_time = end_time - start_time

    # 主进程收集所有进程的局部结果和执行时间
    if rank == 0:
        total_time = np.empty(size, dtype=np.float64)
        for i in range(size):
            if i > 0:
                # 收集执行时间
                comm.Recv(total_time[i:i+1], source=i, tag=i)
                # 收集局部结果
                temp_C_local = np.empty((block_size, block_size), dtype=np.float64)
                comm.Recv(temp_C_local, source=i, tag=i)
                C[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = temp_C_local
            else:
                total_time[0] = execution_time
                C[0:block_size, 0:block_size] = C_local
        average_time = np.mean(total_time)
        print(f"Average execution time: {average_time} seconds")
    else:
        # 其他进程发送执行时间到主进程
        comm.Send(np.array([execution_time], dtype=np.float64), dest=0, tag=rank)
        # 发送局部计算结果到主进程
        comm.Send(C_local, dest=0, tag=rank)

if __name__ == '__main__':
    main()
```

​         对于这个程序，由于其是基于MPI进行完成的，所以我在终端输入了以下指令，为其分配了四个进程

```
# 指令中数字4代表进程数，可进行更改。
mpiexec -n 4 python3 文件路径.py
```

​         然后可在终端直接查看其运行结果。

### 2.使用 multiprocessing 实现多进程计算完成同样的任务。

​         对multiprocessing的实现，代码如下：

```python
import numpy as np
from joblib import Parallel, delayed
import time

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

    average_time = total_time / num_runs

    print(f"Average execution time: {average_time:.4f} seconds")

if __name__ == "__main__":
    main()
```

### 3.比较两者的运行效率。

#### 1.比较运行时间。

​      使用mpi4py进行10000*10000矩阵的平均运算时间结果如下：

<img src="https://www.helloimg.com/i/2024/10/19/67133da9dd273.png" alt="屏幕截图 2024-10-12 192035.png" title="屏幕截图 2024-10-12 192035.png" />

​       平均用时14.1秒。

​       使用multiprocessing进行10000*10000矩阵的运算所用时间结果如下：

<img src="https://www.helloimg.com/i/2024/10/19/67133dac31945.png" alt="屏幕截图 2024-10-16 093422.png" title="屏幕截图 2024-10-16 093422.png" />
​         平均用时13.5秒。

​         由于使用了random来随机化矩阵，这可能会导致相同大小的矩阵的乘法运算量有所波动，这会影响计算的时间，导致计算的时间波动性较强。所以并不能直观地看的出来使用四进程对10000*10000的大型矩阵进行乘法运算的时候，使用mpi4py和使用multiprocessing那个更好一点。

#### 2.比较资源使用情况。

可以使用系统监控工具（如 top、htop 或 nmon）来观察资源使用情况。

```Linux
top
1
```

在这里我选择在终端输入上述指令来打开监控面板。

<img src="https://www.helloimg.com/i/2024/10/19/67133dab6a39e.png" alt="屏幕截图 2024-10-16 093335.png" title="屏幕截图 2024-10-16 093335.png" />

其4进程的运行情况下，CPU占用接近100%，占用相当之高。内存使用情况相对而言没有特别高，有大量的空闲内存和缓存。交换空间完全没有被使用，这也表明系统没有因为内存不足而使用交换空间却运行程序。

<img src="https://www.helloimg.com/i/2024/10/19/67133daadbdda.png" alt="屏幕截图 2024-10-16 093047.png" title="屏幕截图 2024-10-16 093047.png" />

​        而对于multiprocessing来说，可以看出CPU的占用率相较于mpi4py并不是很高啊，内存占用同样不是很高，且相对mpi4py较低。但同样在4线程的情况下，运行时间相差不多。这可能是应为multiprocessing减少了通信的频率，从而减少了因过度通信占用的内存。

1. ###### 扩展测试。

   > [!CAUTION]
   >
   > 更换的进程数必须要能把矩阵内数的个数平分，否则会出现报错的现象。

- 使用不同的进程，来测试不同进程数量下代码正常运行的稳定性。


​        二进程的情况下，mpi4py运行时间用了30秒左右，而相比之下multiprocessing的运行时间为27.7秒，相较于mpi4py只少了不到三秒。

​        四进程情况下，mpi4py运行时间缩短到了14秒左右，multiprocessing的运行时间为13.5秒，两者相差并不是很多。

​        八进程的情况下，mpi4py用时再次减小，只用9秒，而multiprocessing的运行时间相应减少到了12.9秒。

​        十六进程的情况下，mpi4py用时不降反增，用时11秒多，似乎10000*10000的矩阵已经无法满足其性能，而multiprocessing的运行时间为16.4秒。

汇总如图：
<img src="https://www.helloimg.com/i/2024/10/19/67133da6c417a.png" alt="good.png" title="good.png" />

​        如此看来，似乎在矩阵大小为10000*10000的情况下，multiprocessing的计算速度不如mpi4py。当然这也有着一定的局限性，因为设备的性能有限，所测试的数据量不够大，因此无法得出普适性的结论。

​       但单从上图所展示的数据来看，mpi4py无论是在计算时间，还是进程数升高后的稳定性方面，都明显优秀于multiprocessing。

- 使用不同矩阵大小，来比较计算量变化对计算时间的影响。

  对于mpi4py，如图：<img src="https://www.helloimg.com/i/2024/10/19/67133da8c980e.png" alt="re.png" title="re.png" />

  对于multiprocessing，如图：<img src="https://www.helloimg.com/i/2024/10/19/67133da936015.png" alt="tired.png" title="tired.png" />

  由此可见，对于大型矩阵的乘法运算，mpi4py的稳定性明显优于multiprocessing。

## 三.优化方案及其带来的性能提升及对比。

### 1.代码优化。

> [!NOTE]
>
> 为了控制变量，优化后的程序将使用4进程,10000*10000的矩阵大小来进行测试。

#### ①对使用mpi4py的优化：

##### （1）非精度优化

​        在不要求数据精度较高的情况下，我们可以将dtype=np.float64改为dtype=np.float32，来达到降低精度求速度的目的，结果非常成功，计算时间减少到了7.9秒，较之前的减少了一半。

```python
from mpi4py import MPI
import numpy as np
import time

def matrix_multiply(A, B):
    """执行矩阵乘法。"""
    return np.dot(A, B)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  # 获取总进程数

    n = 10000
    # 确保矩阵可以被进程数整除
    assert n % size == 0, "Matrix size must be divisible by the number of processes."

    # 分块大小
    block_size = n // size

    if rank == 0:
        # 主进程初始化矩阵A和B
        A = np.random.rand(n, n).astype(np.float32)
        B = np.random.rand(n, n).astype(np.float32)
        # 初始化用于收集结果的数组
        C = np.empty((n, n), dtype=np.float32)
    else:
        # 其他进程只需要分配空间给局部矩阵
        A_local = np.empty((block_size, n), dtype=np.float32)
        B_local = np.empty((n, block_size), dtype=np.float32)
        # 初始化用于收集结果的数组
        C_local = np.empty((block_size, block_size), dtype=np.float32)

    # 主进程分发矩阵A和B到所有进程
    if rank == 0:
        for i in range(1, size):
            # 确保发送的数组是连续的
            A_slice = A[i*block_size:(i+1)*block_size, :].copy().astype(np.float32)
            B_slice = B[:, i*block_size:(i+1)*block_size].copy().astype(np.float32)
            comm.Send(A_slice, dest=i, tag=11)
            comm.Send(B_slice, dest=i, tag=12)
        # 主进程保留第一块，避免重复计算
        A_local = A[0:block_size, :].astype(np.float32)
        B_local = B[:, 0:block_size].astype(np.float32)
        C_local = np.empty((block_size, block_size), dtype=np.float32)
    else:
        # 其他进程接收矩阵A和B的局部部分
        A_local = np.empty((block_size, n), dtype=np.float32)
        B_local = np.empty((n, block_size), dtype=np.float32)
        comm.Recv(A_local, source=0, tag=11)
        comm.Recv(B_local, source=0, tag=12)

    # 所有进程计算其分配到的矩阵片段的乘法
    start_time = time.perf_counter()
    C_local = matrix_multiply(A_local, B_local)
    end_time = time.perf_counter()

    # 计算执行时间
    execution_time = end_time - start_time

    # 主进程收集所有进程的局部结果和执行时间
    if rank == 0:
        total_time = np.empty(size, dtype=np.float32)
        for i in range(size):
            if i > 0:
                # 收集执行时间
                comm.Recv(total_time[i:i+1], source=i, tag=i)
                # 收集局部结果
                temp_C_local = np.empty((block_size, block_size), dtype=np.float32)
                comm.Recv(temp_C_local, source=i, tag=i)
                C[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = temp_C_local
            else:
                total_time[0] = execution_time
                C[0:block_size, 0:block_size] = C_local
        average_time = np.mean(total_time)
        print(f"Average execution time: {average_time} seconds")
    else:
        # 其他进程发送执行时间到主进程
        comm.Send(np.array([execution_time], dtype=np.float32), dest=0, tag=rank)
        # 发送局部计算结果到主进程
        comm.Send(C_local, dest=0, tag=rank)

if __name__ == '__main__':
    main()
```

##### （2）精度优化

​        但在很多情况下，我们会对程序的精度做出一定的要求，所以我在不改变精度的情况下，对程序进行了算法上的优化。

###### **减少通信次数**：

​        通过一次性发送所有矩阵块，而不是分开发送，减少了通信次数。

###### 使用非阻塞式信道：

​        主进程可以在发送数据的同时还进行计算，减少了不必要的时间浪费。

```Python
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
```

​        通过此次精度优化，成功的把运行时间减少到了10.0秒。

​        若将精度与非精度优化结合，我们便可以将程序的运行时间进一步优化，运行时间减少到了5.4秒，实现了质的飞跃。更换进程数后，程序的运行时间更是得到了进一步的大提升。

​        还有一点，在测试的过程中，我尝试使用非精度的（float32）的方案进行计算，但非常有趣的现象是计算结果出现很多像1.5066332285486077e+23这样非常大的数据。我怀疑是我的算法出现了问题，但在进行实验的时候，我确保了矩阵能被正确分割以及并行通讯的正确，但仍然会出现这种情况。所以我认为这种情况是大概率是因为浮点数的溢出造成的，即中间结果超过了浮点所能表示的最大值，就会发生浮点数溢出。但出现这种情况并不只有这一种可能，还有可能是因为内存过小而被击穿造成的，这一点在我进行较大的矩阵乘法测试时得到了体现，计算的结果有部分是一个负的极大值，这是内存被击穿的体现。在将数据类型改回float64后，就没再发生过此类问题。

​        但这个程序仍然有进一步优化的潜能，在收集结果时，理论可以用MPI_Gather来减少接收操作的次数，代替MPI_Recv来对程序进行优化，并提供一个接收缓冲区列表 recvbuf 和位移列表 displacements确保所有数据被正确接收。但实际上我遇到了无法解决的报错问题，导致优化到此为止。

#### ②对使用multiprocessing的优化：

以下几点，是优化的主要方向。 

（1）**批量处理任务** （2）**减少数据传输** （3）**避免不必要的数据复制**（4）**提高内存使用效率**

​        而这些恰好是joblib的优势所在。joblib库有三大特点：高效存储、并行计算支持和磁盘缓存的优化。在用于存储大型数组时，使用joblib进行数据序列化和反序列化更快。除了本身的优点外，joblib可以还可以使用loky、threading或multiprocessing作为后端来控制任务的执行方式，这也是我选择joblib进行multiprocessing优化的原因。

​         另外，虽然threading也有着较好的数据处理能力，但其多线程的本质使其先天性的不适于此类计算。所以为实现并行计算一般调用多进程的multiprocessing而不是多线程的threading。

以下是优化后的代码：

```python
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
```

​        优化后的代码运行时间有所变化，在10000*10000的矩阵规模下，4进程的运行时间为19.2秒。但如果改变进程数，在8和16进程的运算时，运算时间大幅度减少。

​        虽然是joblib优化，但本质上joblib只是优化了数据存储，其内核还是multiprocessing的多进程运行方案，因此并未将其单列出来，只是把其作为multiprocessing的优化方案在这里显示。

### 2.使用其他方案进行矩阵计算

​        使用scipy进行矩阵计算。

​        scipy建立在numpy之上，是专用于数组的各种计算的，它提供了多维数组操作的扩展功能，有着强大的数组运算能力。那么当只使用scipy和numpy的时候，这个程序的运行速度又会如何呢？

```Python
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
```

​       只能说效果拔群，运算速度很快。由于不分多进程，在计算10000*10000的大型矩阵时，仅用时2.6秒。

<img src="https://www.helloimg.com/i/2024/10/19/67133da90417f.png" alt="scipy1.png" title="scipy1.png" />


​        但明显发现scipy在面对更大的数组是出现了疲软，在矩阵较大时运行时间更是急剧上升。并且在计算20000*20000的矩阵时，程序更是直接就崩溃了，什么都没有输出。而且scipy的也再无从优化，其本身就是一个及其高效的数组计算库，对于较大的数据量的处理较之多进程的其它方案有一定的差距。再优化也只是向着多进程的方向，转而使用mpi4py和multiprocessing，而不再需要scipy。

> [!NOTE]
>
> 以下均为矩阵为10000*10000大小时测试出的数据。

- 可以看到mpi4py在进程数为2时有着最短的运行时间，且遥遥领先。


<img src="https://www.helloimg.com/i/2024/10/19/67133da887043.png" alt="new0.png" title="new0.png" />

​       对于CPU的使用，我把图片拼接在了一起，方便进行对比。（排序为：左上2进程，右上4进程，左下8进程，右下16进程）
<img src="https://www.helloimg.com/i/2024/10/19/67133dad16831.png" alt="拼接.png" title="拼接.png" />

​       可以看出在2进程时CPU的使用率是最高的。所以2进程有着更快地处理速度和最短的完成时间。而对于CPU的使用率，除2进程外不同进程基本持平，而运算速度越快，CPU的占用越高。对于内存，个进程均有着不低的内存占用，但内存的使用仍很健康，且只占用了一部分交换内存空间，没有因频繁使用交换空间造成性能下降。
<img src="https://www.helloimg.com/i/2024/10/19/67133da709aa6.png" alt="CPU占用问题.png" title="CPU占用问题.png" />

​        另外，值得一提的是，在使用mpi4py进行运行时进行CPU的使用监控，发现所有的程序在一开始都是两个个CPU占用飙升而两个基本不怎么使用，但在运行几秒钟后，所有CPU的占用回到了持平。这一点我在查找资料后认为是进程调度策略的原因，本身主进程存在的意义就是先将矩阵切割分发给非主进程进行运算，而这一过程并不需要很高的CPU占用，所以会有部分CPU占用较低。在非阻塞信道的优化后，非主进程会一边处理数据一边将处理完成的数据返回给主进程，由主进程进行收集，所以在分发完数据后，主进程同样开始了数据的收集和处理，这也导致CPU占用升高。

- 对于multiprocessing，其运行速度在进程数较小时远不及mpi4py。如下图：
<img src="https://www.helloimg.com/i/2024/10/19/67133da6c1dfe.png" alt="666.png" title="666.png" />

​          multiprocessing的各进程CPU占用如图。（排序为：左上2进程，右上4进程，左下8进程，右下16进程）

<img src="https://www.helloimg.com/i/2024/10/19/67133dacd1101.png" alt="拼接-1729094466341-6.png" title="拼接-1729094466341-6.png" />

​       明显可以看出multiprocessing对于CPU的占用并没有很高，内存使用情况显示有大量的空闲内存和缓存，但可用内存减少。

​       程序的启动速度非常慢，在开始后的近5秒内，CPU占用基本没有动静，维持在极低的水平，这可能是因为其分割矩阵和分发数据的速度较慢，拖累了整个程序的正常运行。但出奇的是，在低进程数时运行速度低，CPU占用较大的joblib在高进程数进算时大放异彩，随着进程数的增加，其运行的平均时间越来越短，且CPU占用率也大大降低，维持在一个较低的水平。这可能试试因为较多的进程数能减少CPU的空闲时间和更高效的使用内存空间，使多个进程可以同时读取内存和计算。

​       但这不能代表更多的进程数就代表更快的运行速度和更短的运行时间，因为在使用的进程更多时，过多的进程会导致频繁地信息交换，频繁地信息交换会导致占用较大的内存空间，拖慢整个程序的运行。

​       对比两个不同的实现方式，mpi4py 可能能够更有效地利用系统资源，它可以绕过全局解释器锁（GIL），在 Python 中实现真正的并行执行。而 multiprocessing也能做到这一点，因此很难去比较两个不同的实现方式的优劣。

### 3.不同矩阵大小的情况下数据情况及运行性能比较。

<img src="https://www.helloimg.com/i/2024/10/19/6713879995149.png" alt="greet1.png" title="greet1.png" />

<img src="https://www.helloimg.com/i/2024/10/19/67133da6aa4c3.png" alt="10086.png" title="10086.png" />

<img src="https://www.helloimg.com/i/2024/10/19/67133da6d1332.png" alt="333.png" title="333.png" />

<img src="https://www.helloimg.com/i/2024/10/19/67138abbb419a.png" alt="final.png" title="final.png" />

​       二者的性能略有差异，但都呈现随着矩阵大小增大，运算时间变长的趋势。且可以很明显的看出来，在进程数较大的情况下，joblib优化后的multiprocessing程序明显要优于mpi4py，但在进程数较小时，其运行时间也相应的长于mpi4py，只能说各有优劣。

## 四.实验过程中遇到的问题及解决方案。

在实验过程中遇到了一系列问题，问题如下（附解决方法）

- 无法正确安装multiprocessing库

- 使用mpi4py时无法正常使用，出现Open MPI和MPI混乱的情况。

  出现错误的原因在于一开始安装mpi4py时自动配置了MPI，而在查阅资料后我又安装了Open MPI，导致两者混乱。

  解决方法是删除Open MPI。

  ```Linux
  sudo apt-get remove openmpi-bin libopenmpi-dev
  ```

- 本来能正常运行的程序，在隔了一个晚上后，无法计算出结果。

  解决方法为恢复了提前拍摄好的快照。


参考资料：

​        壹--[cscode环境配置]([在VScode中配置Python开发环境_vscode配置python开发环境-CSDN博客](https://blog.csdn.net/weixin_43737995/article/details/125690015?ops_request_misc=%7B%22request%5Fid%22%3A%2267236699-1E27-451A-B30B-52EC49C1964F%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=67236699-1E27-451A-B30B-52EC49C1964F&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-125690015-null-null.142^v100^pc_search_result_base1&utm_term=vscode python环境配置&spm=1018.2226.3001.4187))

​        贰--[安装multiprocessing库](https://blog.csdn.net/lin873/article/details/134202279?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522F5B043C5-B9A1-4D67-AF51-460206F1A3DE%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=F5B043C5-B9A1-4D67-AF51-460206F1A3DE&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-134202279-null-null.142^v100^pc_search_result_base1&utm_term=multiprocessing%E5%BA%93%E5%AE%89%E8%A3%85&spm=1018.2226.3001.4187)        

​        叁--[Python多进程库multiprocessing中进程池Pool类的使用](https://blog.csdn.net/jinping_shi/article/details/52433867?ops_request_misc=&request_id=&biz_id=102&utm_term=python%E4%BD%BF%E7%94%A8multiprocessing%E8%BF%9B%E8%A1%8C%E5%A4%A7%E5%9E%8B%E7%9F%A9%E9%98%B5%E4%B9%98&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-5-52433867.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187)

​        肆--[Multiprocessing 让你的多核计算机发挥真正潜力 Python]([【莫烦Python】Multiprocessing 让你的多核计算机发挥真正潜力 Python-希望硬币无限多-稍后再看-哔哩哔哩视频 (bilibili.com)](https://www.bilibili.com/list/watchlater?oid=16944405&bvid=BV1jW411Y7pv&spm_id_from=333.1007.top_right_bar_window_view_later.content.click))

​        伍--[高性能计算的矩阵乘法优化 - Python +MPI的实现](https://blog.csdn.net/rizero/article/details/129775285?ops_request_misc=%257B%2522request%255Fid%2522%253A%25224D9E1B24-BEB7-45EC-B8BD-FF397459D973%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=4D9E1B24-BEB7-45EC-B8BD-FF397459D973&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-129775285-null-null.142^v100^pc_search_result_base1&utm_term=python%E4%BD%BF%E7%94%A8mpi4py%E8%BF%9B%E8%A1%8C%E5%A4%A7%E5%9E%8B%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95&spm=1018.2226.3001.4187)  

​        陆--[python | joblib，一个强大的 Python 库！]([python | joblib，一个强大的 Python 库！-CSDN博客](https://blog.csdn.net/csdn_xmj/article/details/138958218?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-0-138958218-blog-136809087.235^v43^pc_blog_bottom_relevance_base8&spm=1001.2101.3001.4242.1&utm_relevant_index=1))

​        柒--[Python Joblib 工具使用]([02-Joblib 并行计算_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1AF4m157WQ?spm_id_from=333.788.player.switch&vd_source=2300dac6a683978c53acec126f8ccb4d&p=2))

​        捌--[CSV文件的基本读写操作：深入解析与Python实践](https://blog.csdn.net/weixin_43856625/article/details/141455020?ops_request_misc=&request_id=&biz_id=102&utm_term=csv%E6%96%87%E4%BB%B6&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-141455020.142^v100^pc_search_result_base1&spm=1018.2226.3001.4187)

## 特别鸣谢

***回归天空（[SXP-Simon](https://github.com/SXP-Simon)） Kimi***  ***佬鸽([olddove-laoge](https://github.com/olddove-laoge))***  ***全体NCUSCC招新群群友***

**此排名不分先后，按作者想起的顺序进行排列。**

