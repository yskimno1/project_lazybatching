import math
import random
import os
import sys
from random import randrange

experiment_time      = 10000.0  # in millisecond
interval_list        = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0] # in millisecond
num_max_request      = 1000

def nextTime(num_arrival):
    return -math.log(1.0 - random.random()) / num_arrival

def generate_set(interval):
    num_arrival = (float(1) / interval)

    cur_cycle = 1000.0
    new_set   = set()

    while (cur_cycle < experiment_time and len(new_set) <= num_max_request):
        new_set      = new_set | {(cur_cycle)}
        nxt_seed     = nextTime(num_arrival)
        cur_cycle    = cur_cycle + nxt_seed
        #print(cur_cycle)

    return new_set

def to_string_list(l):
    #l = [str(elem) for elem in l]
    l = ["{:.1f}".format(elem) for elem in l]
    l = ','.join(l)
    return l

for interval_i in range(len(interval_list)):
    generated_set = generate_set(interval_list[interval_i])
    generated_set = generated_set
    generated_list = list(generated_set)
    generated_list.sort()
    
    if(len(generated_list) > 1000):
        generated_list = generated_list[:1000]

    request_cycle = to_string_list(generated_list)
    print(request_cycle)

    f = open("testcase_"+str(interval_list[interval_i])+".txt", 'w')
    f.write(request_cycle)

 
#      f = open("scripts/"+benchmark[b_i]+"/"+batch_policy[p_i]+"/script_"+str(interval_i)+".sh" , 'w')
#      f.write("#! /bin/bash\n")
#      f.write("#PBS -N "+cur_change+"\n")
#      f.write("#PBS -q batch\n")
#      f.write("cd /home/jjeong/2019/batch_sim/bin\n")
#      
#      f.write(" ".join(output))
     
      #print("#! /bin/bash")
      #print("PBS -N job_0")
      #print("PBS -q batch")
      #print("cd /home/jjeong/2019/batch_sim/bin")
      #
      #print(" ".join(output))