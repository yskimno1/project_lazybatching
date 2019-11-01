#/usr/bin/bash

PROJECT_PATH=$HOME/XBatch/XBatch/Project
ARR_INTERVAL=(1.25 2.5 5.0 5.5 6.0 6.5 7.0 7.5 8.5 10.0 20.0 40.0 80.0)
ARR_BATCH=(16)


echo "-----------X-BATCHING-----------"

for i in ${ARR_INTERVAL[@]}
do
    for j in ${ARR_BATCH[@]}
    do
        grep latency $PROJECT_PATH/result_anybatch_new/result_anybatch_interval-$i-batch-$j > grepdata
        echo "interval ${i} batch ${j}"
        python parser.py        
    done
done

echo "--------------------------------"
echo "--------DYNAMIC BATCHING--------"
for i in ${ARR_INTERVAL[@]}
do
    for j in ${ARR_BATCH[@]}
    do
        grep latency $PROJECT_PATH/result_dynamic_new/result_dynamic_interval-$i-batch-$j > grepdata
        echo "interval ${i} batch ${j}"
        python parser.py        
    done
done

echo "--------------------------------"


# for k in ${ARR_BATCH_NEW[@]}
# do
#     grep latency $PROJECT_PATH/result_191101/batch_${k}_exec_time_4.txt > grepdata
#     echo "batch ${k}"
#     python parser.py    
# done


# grep latency dynamic_0.25ms > grepdata 
# echo 'interval 0.25'
# python parser.py
# grep latency dynamic_0.5ms > grepdata 
# echo 'interval 0.5'
# python parser.py
# grep latency dynamic_1ms > grepdata
# echo 'interval 1'
# python parser.py
# grep latency dynamic_1.1ms > grepdata
# echo 'interval 1.1'
# python parser.py
# grep latency dynamic_1.2ms > grepdata
# echo 'interval 1.2'
# python parser.py
# grep latency dynamic_1.3ms > grepdata
# echo 'interval 1.3'
# python parser.py
# grep latency dynamic_1.4ms > grepdata
# echo 'interval 1.4'
# python parser.py
# grep latency dynamic_1.5ms > grepdata
# echo 'interval 1.5'
# python parser.py
# grep latency dynamic_1.6ms > grepdata
# echo 'interval 1.6'
# python parser.py
# grep latency dynamic_1.7ms > grepdata
# echo 'interval 1.7'
# python parser.py
# grep latency dynamic_1.8ms > grepdata
# echo 'interval 1.8'
# python parser.py
# grep latency dynamic_1.9ms > grepdata
# echo 'interval 1.9'
# python parser.py

# grep latency dynamic_2ms > grepdata
# echo 'interval 2'
# python parser.py

# grep latency dynamic_3ms > grepdata
# echo 'interval 3'
# python parser.py

# grep latency dynamic_5ms > grepdata
# echo 'interval 5'
# python parser.py

# grep latency dynamic_10ms > grepdata
# echo 'interval 10'
# python parser.py

# grep latency dynamic_10ms > grepdata
# echo 'interval 10'
# python parser.py

# grep latency dynamic_20ms > grepdata
# echo 'interval 20'
# python parser.py

# grep latency dynamic_30ms > grepdata
# echo 'interval 30'
# python parser.py

# grep latency dynamic_50ms > grepdata
# echo 'interval 50'
# python parser.py