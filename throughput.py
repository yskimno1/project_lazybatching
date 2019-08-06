import os
import sys

f_read = open(sys.argv[1], 'r')

for line in f_read:
    if 'no.999' in line:
        print(sys.argv[1], "last : ", line)
