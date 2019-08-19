#!/bin/sh

python generate_poisson.py
./anybatch testcase_0.1.txt > anybatch_poisson_0.1ms
./anybatch testcase_0.25.txt > anybatch_poisson_0.25ms
./anybatch testcase_0.5.txt > anybatch_poisson_0.5ms
./anybatch testcase_1.0.txt > anybatch_poisson_1.0ms
./anybatch testcase_2.0.txt > anybatch_poisson_2.0ms
./anybatch testcase_4.0.txt > anybatch_poisson_4.0ms
./anybatch testcase_8.0.txt > anybatch_poisson_8.0ms
./anybatch testcase_16.0.txt > anybatch_poisson_16.0ms
./anybatch testcase_32.0.txt > anybatch_poisson_32.0ms
./anybatch testcase_64.0.txt > anybatch_poisson_64.0ms

# python input_generator.py 1000 0.05
# ./anybatch requests.txt > anybatch_0.05ms

# python input_generator.py 1000 0.1
# ./anybatch requests.txt > anybatch_0.1ms

# python input_generator.py 1000 0.25
# ./anybatch requests.txt > anybatch_0.25ms

# python input_generator.py 1000 0.5
# ./anybatch requests.txt > anybatch_0.5ms

# python input_generator.py 1000 1
# ./anybatch requests.txt > anybatch_1ms

# python input_generator.py 1000 1.1
# ./anybatch requests.txt > anybatch_1.1ms

# python input_generator.py 1000 1.2
# ./anybatch requests.txt > anybatch_1.2ms

# python input_generator.py 1000 1.3
# ./anybatch requests.txt > anybatch_1.3ms

# python input_generator.py 1000 1.4
# ./anybatch requests.txt > anybatch_1.4ms

# python input_generator.py 1000 1.5
# ./anybatch requests.txt > anybatch_1.5ms

# python input_generator.py 1000 1.6
# ./anybatch requests.txt > anybatch_1.6ms

# python input_generator.py 1000 1.7
# ./anybatch requests.txt > anybatch_1.7ms

# python input_generator.py 1000 1.8
# ./anybatch requests.txt > anybatch_1.8ms

# python input_generator.py 1000 1.9
# ./anybatch requests.txt > anybatch_1.9ms

# python input_generator.py 1000 2
# ./anybatch requests.txt > anybatch_2ms

# python input_generator.py 1000 3
# ./anybatch requests.txt > anybatch_3ms

# python input_generator.py 1000 5
# ./anybatch requests.txt > anybatch_5ms

# python input_generator.py 1000 10
# ./anybatch requests.txt > anybatch_10ms

# python input_generator.py 1000 20
# ./anybatch requests.txt > anybatch_20ms

# python input_generator.py 1000 30
# ./anybatch requests.txt > anybatch_30ms

# python input_generator.py 1000 50
# ./anybatch requests.txt > anybatch_50ms