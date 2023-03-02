import torch
import time

###CPU
start_time = time.time()
a = torch.ones(40000,40000)
for i in range(100):
    a += a
    if i % 10 == 0:
        print(i)
elapsed_time = time.time() - start_time

print('CPU time = ',elapsed_time)

###GPU
start_time = time.time()
b = torch.ones(40000,40000).cuda()
for i in range(1000):
    b += b
    if i % 10 == 0:
        print(i)
elapsed_time = time.time() - start_time

print('GPU time = ',elapsed_time)
