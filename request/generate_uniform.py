import os
import sys
import argparse

parser = argparse.ArgumentParser(description='generate requests.')
parser.add_argument('number', type=int, help='number of requests')
parser.add_argument('interval', type=float, help='interval between requests')

args = parser.parse_args()
# print(args.number)
s = "uniform_" + str(args.interval) + ".txt"

f = open(s, 'w')
for i in range(args.number):
  req = (1000 + args.interval * i)
  req = round(req, 4);
  if(i == args.number-1):
      data = str(req)
  else:
      data = str(req) + ","
  f.write(data)
  
f.close()
