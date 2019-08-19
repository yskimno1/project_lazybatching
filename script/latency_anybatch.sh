#/usr/bin/bash

grep latency anybatch_poisson_0.1ms > grepdata
echo 'interval 0.1'
python parser.py
grep latency anybatch_poisson_0.25ms > grepdata
echo 'interval 0.25'
python parser.py
grep latency anybatch_poisson_0.5ms > grepdata
echo 'interval 0.5'
python parser.py
grep latency anybatch_poisson_1.0ms > grepdata
echo 'interval 1'
python parser.py
grep latency anybatch_poisson_2.0ms > grepdata
echo 'interval 2'
python parser.py
grep latency anybatch_poisson_4.0ms > grepdata
echo 'interval 4'
python parser.py
grep latency anybatch_poisson_8.0ms > grepdata
echo 'interval 8'
python parser.py
grep latency anybatch_poisson_16.0ms > grepdata
echo 'interval 16'
python parser.py
grep latency anybatch_poisson_32.0ms > grepdata
echo 'interval 32'
python parser.py
grep latency anybatch_poisson_64.0ms > grepdata
echo 'interval 64'
python parser.py

# grep latency anybatch_0.05ms > grepdata
# echo 'interval 0.05'
# python parser.py

# grep latency anybatch_0.1ms > grepdata 
# echo 'interval 0.1'
# python parser.py

# grep latency anybatch_0.25ms > grepdata 
# echo 'interval 0.25'
# python parser.py

# grep latency anybatch_0.5ms > grepdata 
# echo 'interval 0.5'
# python parser.py

# grep latency anybatch_1ms > grepdata
# echo 'interval 1'
# python parser.py

# grep latency anybatch_1.1ms > grepdata
# echo 'interval 1.1'
# python parser.py
# grep latency anybatch_1.2ms > grepdata
# echo 'interval 1.2'
# python parser.py
# grep latency anybatch_1.3ms > grepdata
# echo 'interval 1.3'
# python parser.py
# grep latency anybatch_1.4ms > grepdata
# echo 'interval 1.4'
# python parser.py
# grep latency anybatch_1.5ms > grepdata
# echo 'interval 1.5'
# python parser.py
# grep latency anybatch_1.6ms > grepdata
# echo 'interval 1.6'
# python parser.py
# grep latency anybatch_1.7ms > grepdata
# echo 'interval 1.7'
# python parser.py
# grep latency anybatch_1.8ms > grepdata
# echo 'interval 1.8'
# python parser.py
# grep latency anybatch_1.9ms > grepdata
# echo 'interval 1.9'
# python parser.py

# grep latency anybatch_2ms > grepdata
# echo 'interval 2'
# python parser.py

# grep latency anybatch_3ms > grepdata
# echo 'interval 3'
# python parser.py

# grep latency anybatch_5ms > grepdata
# echo 'interval 5'
# python parser.py

# grep latency anybatch_10ms > grepdata
# echo 'interval 10'
# python parser.py

# grep latency anybatch_20ms > grepdata
# echo 'interval 20'
# python parser.py

# grep latency anybatch_30ms > grepdata
# echo 'interval 30'
# python parser.py

# grep latency anybatch_50ms > grepdata
# echo 'interval 50'
# python parser.py

# # grep latency anybatch_120ms > grepdata
# # echo 'interval 120'
# # python parser.py

