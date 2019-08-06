#!/bin/bash
python input_generator.py 1000 0.05
./anybatch requests.txt >serial_0.05ms

python input_generator.py 1000 0.1
./anybatch requests.txt >serial_0.1ms

python input_generator.py 1000 0.25
./anybatch requests.txt >serial_0.25ms

python input_generator.py 1000 0.5
./anybatch requests.txt >serial_0.5ms

python input_generator.py 1000 1
./anybatch requests.txt >serial_1ms

python input_generator.py 1000 1.1
./anybatch requests.txt >serial_1.1ms

python input_generator.py 1000 1.2
./anybatch requests.txt >serial_1.2ms

python input_generator.py 1000 1.3
./anybatch requests.txt >serial_1.3ms

python input_generator.py 1000 1.4
./anybatch requests.txt > serial_1.4ms

python input_generator.py 1000 1.5
./anybatch requests.txt > serial_1.5ms

python input_generator.py 1000 1.6
./anybatch requests.txt > serial_1.6ms

python input_generator.py 1000 1.7
./anybatch requests.txt > serial_1.7ms

python input_generator.py 1000 1.8
./anybatch requests.txt > serial_1.8ms

python input_generator.py 1000 1.9
./anybatch requests.txt > serial_1.9ms

python input_generator.py 1000 2
./anybatch requests.txt > serial_2ms

python input_generator.py 1000 3
./anybatch requests.txt > serial_3ms

python input_generator.py 1000 5
./anybatch requests.txt > serial_5ms

python input_generator.py 1000 10
./anybatch requests.txt > serial_10ms

python input_generator.py 1000 20
./anybatch requests.txt > serial_20ms

python input_generator.py 1000 30
./anybatch requests.txt > serial_30ms

python input_generator.py 1000 50
./anybatch requests.txt > serial_50ms