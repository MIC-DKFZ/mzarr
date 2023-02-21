import numpy as np
import mzz
import time
import os

warmup_repetions = 100
repetitions = 1000

total_time = 0
for r in range(repetitions):
    array = np.random.rand(1000, 1000).astype(np.float32)
    start_time = time.time()
    np.save("tmp.npy", array)
    if r > warmup_repetions:
        total_time += time.time() - start_time
os.remove("tmp.npy")

print("npy total time (shape: {}): {}s".format(array.shape, total_time))

total_time = 0
for r in range(repetitions):
    array = np.random.rand(1000, 1000).astype(np.float32)
    start_time = time.time()
    mzz.Mzz(array).save("tmp.mzz")
    if r > warmup_repetions:
        total_time += time.time() - start_time
os.remove("tmp.mzz")

print("mzz total time (shape: {}): {}s".format(array.shape, total_time))


total_time = 0
for r in range(repetitions):
    array = np.random.rand(10000, 10000).astype(np.float32)
    start_time = time.time()
    np.save("tmp.npy", array)
    if r > warmup_repetions:
        total_time += time.time() - start_time
os.remove("tmp.npy")

print("npy total time (shape: {}): {}s".format(array.shape, total_time))

total_time = 0
for r in range(repetitions):
    array = np.random.rand(10000, 10000).astype(np.float32)
    start_time = time.time()
    mzz.Mzz(array).save("tmp.mzz")
    if r > warmup_repetions:
        total_time += time.time() - start_time
os.remove("tmp.mzz")

print("mzz total time (shape: {}): {}s".format(array.shape, total_time))