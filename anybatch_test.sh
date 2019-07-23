#!/bin/sh

python input_generator.py 1000 0.001
./a.out requests.txt > batch_1_1ms

python input_generator.py 1000 0.005
./a.out requests.txt > batch_1_5ms

python input_generator.py 1000 0.01
./a.out requests.txt > batch_1_10ms

python input_generator.py 1000 0.02
./a.out requests.txt > batch_1_20ms

python input_generator.py 1000 0.03
./a.out requests.txt > batch_1_30ms


