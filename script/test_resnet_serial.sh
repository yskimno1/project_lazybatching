#!/bin/bash

echo 'you had have changed layer.h!'

./build.sh
echo '----------------\n'
echo 'build done\n'
echo '----------------\n'
./serial_test.sh
echo '----------------\n'
echo 'test done\n'
echo '----------------\n'

./throughput_serial.sh
echo '----------------\n'
echo 'throughput done\n'
echo '----------------\n'

./latency_serial.sh
echo '-----------------\n'
echo 'latency done\n'
