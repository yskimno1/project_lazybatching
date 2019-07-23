import os
import sys

f_read = open('grepdata', 'r')

sum = 0
while True:
    line = f_read.readline()
    if not line: break

    latency = line.split(':')[1]
    latency = latency.split('\n')[0]
    latency = latency.split(' ')[1]
    # print(latency)
    sum += float(latency)

latency = sum / 1000
print("latency : ", latency)