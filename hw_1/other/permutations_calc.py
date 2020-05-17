

import math

k_vals = list([10, 13, 15, 16, 17, 20, 21, 24])
calc_speed = 2**30

pos_paths = []
pos_paths_log = []
calc_time_s = []

path_calc = lambda k : math.factorial(k)

with open("out.txt", "w") as f: 
	for k in k_vals:
		pos_paths.append(path_calc(k))
		pos_paths_log.append(math.log2(pos_paths[-1]))
		calc_time_s.append(pos_paths[-1] / calc_speed)
		f.write(f"k = {k} , possible paths = {pos_paths[-1]} , \
			log2 = {pos_paths_log[-1]}, calc time = {round(calc_time_s[-1],10)} [s]\n")




