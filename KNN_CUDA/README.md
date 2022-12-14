# KNN_CUDA

+ ref: [kNN-CUDA](https://github.com/vincentfpgarcia/kNN-CUDA)
+ ref: [pytorch knn cuda](https://github.com/chrischoy/pytorch_knn_cuda)
+ author: [sli@mail.bnu.edu.cn](sli@mail.bnu.edu.cn)
+ modify: [zhanhz20@mails.tsinghua.edu.cn](zhanhz20@mails.tsinghua.edu.cn)

#### Modifications 
+ Aten support
+ pytorch v1.0+ support
+ pytorch c++ extention 

#### Performance

+ dim   = 5
+ k     = 100
+ ref   = 224
+ query = 224
+ Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
+ NVIDIA GeForce 940MX

| Loop   | sklearn | CUDA    | Memory   |
| :---:  | :---:   | :---:   | :---:    |
| 100    | 2.34 ms | 0.06 ms | 652/1024 |
| 1000   | 2.30 ms | 1.40 ms | 652/1024 |


#### Usage

```python
import torch

# Make sure your CUDA is available.
assert torch.cuda.is_available()

from knn_cuda import KNN
"""
if transpose_mode is True, 
    ref   is Tensor [bs x nr x dim]
    query is Tensor [bs x nq x dim]
    
    return 
        dist is Tensor [bs x nq x k]
        indx is Tensor [bs x nq x k]
else
    ref   is Tensor [bs x dim x nr]
    query is Tensor [bs x dim x nq]
    
    return 
        dist is Tensor [bs x k x nq]
        indx is Tensor [bs x k x nq]
"""

knn = KNN(k=10, transpose_mode=True)

ref = torch.rand(32, 1000, 5).cuda()
query = torch.rand(32, 50, 5).cuda()

dist, indx = knn(ref, query)  # 32 x 50 x 10
```
