#!/bin/bash

PROJECT_PATH=$HOME/XBatch/XBatch/Project

$PROJECT_PATH/build/build_dynamic.sh
ARR_INTERVAL=(5.5 6.0 6.5 7.0)
# ARR_INTERVAL=(2.5 5.0 10.0 20.0 40.0 80.0)
for i in ${ARR_INTERVAL[@]}
do
    $PROJECT_PATH/dynamic $PROJECT_PATH/request/uniform_$i.txt > result_dynamic_new/result_dynamic_interval-$i-batch-16
done


$PROJECT_PATH/build/build.sh

for i in ${ARR_INTERVAL[@]}
do
    $PROJECT_PATH/anybatch $PROJECT_PATH/request/uniform_$i.txt > result_anybatch_new/result_anybatch_interval-$i-batch-16
done