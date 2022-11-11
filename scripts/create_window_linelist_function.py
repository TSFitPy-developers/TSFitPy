import os
from os import path as os_path
import glob
import time
import numpy as np

def create_window_linelist(seg_begins, seg_ends, old_path_name, new_path_name, molecules_flag, start, stop, lbl=False):
	line_list_path = old_path_name
	line_list_files_draft = []
	line_list_files_draft.extend([i for i in glob.glob(os_path.join(line_list_path, "*")) if not i.endswith(".txt")])
	#print(line_list_files)

	if not lbl:
		if not os.path.exists(new_path_name):
			os.makedirs(new_path_name)
		else:
			print("Trimmed files exist already")
			return
	else:
		if not os.path.exists(new_path_name):
			os.makedirs(new_path_name)

	line_list_files = []
	for i in range(len(line_list_files_draft)):
		with open(line_list_files_draft[i]) as fp:
			line = fp.readline()
			fields = line.strip().split()
			sep = '.'
			element = fields[0] + fields[1]
			elements = element.split(sep, 1)[0]
			if len(elements) > 3 and molecules_flag == 'True':
				line_list_files.append(line_list_files_draft[i])
			elif len(elements) <= 3:
				line_list_files.append(line_list_files_draft[i])
		fp.close()

	for i in range(len(line_list_files)):
		count_elements = 0
		with open(line_list_files[i]) as fp:
			line = fp.readline()
			while line:
				fields = line.strip().split()
				if fields[0][0] == "'":
					count_elements += 1
					line = fp.readline()
				line = fp.readline()
		fp.close()
		#count_lines = [[0 for j in range(len(seg_ends))] for i in range(count_elements)]
		count_lines = [0 for i in range(count_elements)]
		#print(count_lines)
		element_names = ['x' for i in range(count_elements)]
#		for j in range(len(seg_ends)):
		k = 0
		with open(line_list_files[i]) as fp:
			line = fp.readline()
			while line:
				fields = line.strip().split()
				if fields[0][0] == "'":
					element_names[k] = fields[0]+fields[1]
					if element_names[k] == "'01.000000'":
						count_lines[k] = int(fields[3])
					k += 1
					line = fp.readline()
				else:
					for j in range(start, stop, 1):
						if float(fields[0]) <= seg_ends[j] and float(fields[0]) >= seg_begins[j]:
							count_lines[k-1] += 1
				line = fp.readline()
		fp.close()
		new_linelist = os_path.join("{}".format(new_path_name), "linelist-{}.bsyn".format(i))
		g = open(new_linelist, "w")
		flag_lines = 0
		k = 0
		with open(line_list_files[i]) as fp:
			line = fp.readline()
			while line:
				fields = line.strip().split()
				if fields[0][0] == "'" and fields[0]+fields[1] == element_names[k] and len(fields[0]) > 1:
					if count_lines[k] != 0:
						g.write("{} {}  {}   {}\n".format(fields[0], fields[1], fields[2], count_lines[k]))
					line = fp.readline()
					if count_lines[k] != 0:
						g.write("{}\n".format(line.strip()))
					k+=1
					#flag_header += 1
				elif fields[0][0] == "'" and fields[0]+fields[1] == element_names[k]:
					if count_lines[k] != 0:
						g.write("{}   {}            {}    {}        {}\n".format(fields[0], fields[1], fields[2], fields[3], count_lines[k]))
					line = fp.readline()
					if count_lines[k] != 0:
						g.write("{}\n".format(line.strip()))
					k+=1
					#flag_header += 1
				#elif fields[0][0] == "'" and fields[0]+fields[1] == element_names[k] and len(fields[0]) > 1 and flag_header != 0:
				#	line = fp.readline()
				#	k+=1
				#	flag_header+=1
				#elif fields[0][0] == "'" and fields[0]+fields[1] == element_names[k] and flag_header != 0:
				#	line = fp.readline()
				#	k+=1
				#	flag_header+=1
				else:
					for j in range(start, stop, 1):
						if float(fields[0]) <= seg_ends[j] and float(fields[0]) >= seg_begins[j]:
					#print(line)
							flag_lines = 1
							if count_lines[k-1] != 0:
								g.write("{} \n".format(line.strip()))
					if element_names[k-1] == "'01.000000'":# and float(fields[0]) <= np.max(seg_ends) and float(fields[0]) >= np.min(seg_begins):
						flag_lines = 1
						if count_lines[k-1] != 0:
							g.write("{} \n".format(line.strip()))
				line = fp.readline()
		fp.close()
		g.close()
		if flag_lines == 0:
			os.system("rm {}".format(new_linelist))
		flag_lines = 0

