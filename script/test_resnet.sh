#!/bin/sh

./build.sh
./build_dynamic.sh
echo '----------------\n'
echo 'build done\n'
echo '----------------\n'
./anybatch_test.sh
./dynamic_test.sh
echo '----------------\n'
echo 'test done\n'
echo '----------------\n'
./throughput_anybatch.sh
./throughput_dynamic.sh
echo '----------------\n'
echo 'throughput done\n'
echo '----------------\n'
./latency_anybatch.sh
./latency_dynamic.sh
