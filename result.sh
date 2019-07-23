#/usr/bin/bash

grep latency batch_1_1ms > grepdata
echo 'interval 1'
python parser.py

grep latency batch_1_5ms > grepdata
echo 'interval 5'
python parser.py

grep latency batch_1_10ms > grepdata 
echo 'interval 10'
python parser.py

grep latency batch_1_20ms > grepdata 
echo 'interval 20'
python parser.py

grep latency batch_1_30ms > grepdata 
echo 'interval 30'
python parser.py
