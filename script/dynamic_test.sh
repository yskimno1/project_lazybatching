#!/bin/sh

python generate_poisson.py
./anybatch testcase_0.1.txt > dynamic_poisson_0.1ms
./anybatch testcase_0.25.txt > dynamic_poisson_0.25ms
./anybatch testcase_0.5.txt > dynamic_poisson_0.5ms
./anybatch testcase_1.0.txt > dynamic_poisson_1.0ms
./anybatch testcase_2.0.txt > dynamic_poisson_2.0ms
./anybatch testcase_4.0.txt > dynamic_poisson_4.0ms
./anybatch testcase_8.0.txt > dynamic_poisson_8.0ms
./anybatch testcase_16.0.txt > dynamic_poisson_16.0ms
./anybatch testcase_32.0.txt > dynamic_poisson_32.0ms
./anybatch testcase_64.0.txt > dynamic_poisson_64.0ms

# python input_generator.py 1000 0.05
# ./dynamic requests.txt > dynamic_0.05ms

# python input_generator.py 1000 0.1
# ./dynamic requests.txt > dynamic_0.1ms

# python input_generator.py 1000 0.25
# ./dynamic requests.txt > dynamic_0.25ms

# python input_generator.py 1000 0.5
# ./dynamic requests.txt > dynamic_0.5ms

# python input_generator.py 1000 1
# ./dynamic requests.txt > dynamic_1ms

# python input_generator.py 1000 1.1
# ./dynamic requests.txt > dynamic_1.1ms

# python input_generator.py 1000 1.2
# ./dynamic requests.txt > dynamic_1.2ms

# python input_generator.py 1000 1.3
# ./dynamic requests.txt > dynamic_1.3ms

# python input_generator.py 1000 1.4
# ./dynamic requests.txt > dynamic_1.4ms

# python input_generator.py 1000 1.5
# ./dynamic requests.txt > dynamic_1.5ms

# python input_generator.py 1000 1.6
# ./dynamic requests.txt > dynamic_1.6ms

# python input_generator.py 1000 1.7
# ./dynamic requests.txt > dynamic_1.7ms

# python input_generator.py 1000 1.8
# ./dynamic requests.txt > dynamic_1.8ms

# python input_generator.py 1000 1.9
# ./dynamic requests.txt > dynamic_1.9ms

# python input_generator.py 1000 2
# ./dynamic requests.txt > dynamic_2ms

# python input_generator.py 1000 3
# ./dynamic requests.txt > dynamic_3ms

# python input_generator.py 1000 5
# ./dynamic requests.txt > dynamic_5ms

# python input_generator.py 1000 10
# ./dynamic requests.txt > dynamic_10ms

# python input_generator.py 1000 20
# ./dynamic requests.txt > dynamic_20ms

# python input_generator.py 1000 30
# ./dynamic requests.txt > dynamic_30ms

# python input_generator.py 1000 50
# ./dynamic requests.txt > dynamic_50ms
