from __future__ import annotations
import os
from os import path as os_path
import glob
import numpy as np
import typing


def create_window_linelist(seg_begins: list[float], seg_ends: list[float], old_path_name: str, new_path_name: str,
						   molecules_flag: str, start: int, stop: int, lbl=False):
	line_list_path: str = old_path_name
	line_list_files_draft: list = []
	line_list_files_draft.extend([i for i in glob.glob(os_path.join(line_list_path, "*")) if not i.endswith(".txt")])
	#print(line_list_files)

	segment_to_use_begins: np.ndarray = np.asarray(seg_begins)[start:stop]
	segment_to_use_ends: np.ndarray = np.asarray(seg_ends)[start:stop]

	segment_index_order: np.ndarray = np.argsort(segment_to_use_begins)
	segment_to_use_begins: np.ndarray = segment_to_use_begins[segment_index_order]
	segment_to_use_ends: np.ndarray = segment_to_use_ends[segment_index_order]
	segment_min_wavelength: float = np.min(segment_to_use_begins)
	segment_max_wavelength: float = np.max(segment_to_use_ends)

	if not lbl:
		if not os.path.exists(new_path_name):
			os.makedirs(new_path_name)
		else:
			print("Trimmed files exist already")
			#return
			print("Overwriting old file list")
	else:
		if not os.path.exists(new_path_name):
			os.makedirs(new_path_name)

	line_list_files: list = []
	# opens each file, reads first row, if it is long enough then it is molecule. If fitting molecules, then add to the line list, otherwise ignore molecules FAST
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

	for i, line_list_file in enumerate(line_list_files):
		new_linelist: str = os_path.join(f"{new_path_name}", f"linelist-{i}.bsyn")
		with open(new_linelist, "w") as new_file_to_write:
			with open(line_list_file) as fp:
				lines_file: list[str] = fp.readlines()
				file_not_empty: bool = False  # deletes the file if it was empty (no elements there)
				all_lines_to_write: list = []
				line_number_read_for_element: int = 0
				line_number_read_file: int = 0
				total_lines_in_file: int = len(lines_file)
				while line_number_read_file < total_lines_in_file:		# go through all line
					line: str = lines_file[line_number_read_file]
					fields: list[str] = line.strip().split()

					# it means this is an element
					if len(all_lines_to_write) > 0:  # if there was an element before with segments, then write them first
						write_lines(all_lines_to_write, elem_line_1_to_save, elem_line_2_to_save,
									new_file_to_write)
						file_not_empty = True
						all_lines_to_write: list = []
					element_name = f"{fields[0]}{fields[1]}"

					if element_name == "'01.000000'":		# find out whether it is hydrogen
						hydrogen_element: bool = True
					else:
						hydrogen_element: bool = False
					if len(fields[0]) > 1:		# save the first two lines of an element for the future
						elem_line_1_to_save: str = f"{fields[0]} {fields[1]}  {fields[2]}"  # first line of the element
						number_of_lines_element: int = int(fields[3])
					else:
						elem_line_1_to_save: str = f"{fields[0]}   {fields[1]}            {fields[2]}    {fields[3]}"
						number_of_lines_element: int = int(fields[4])
					line_number_read_file += 1
					line: str = lines_file[line_number_read_file]
					elem_line_2_to_save: str = f"{line.strip()}\n"  # second line of the element

					# now we are reading the element's wavelength and stuff
					line_number_read_file += 1
					# lines_for_element = lines_file[line_number_read_file:number_of_lines_element+line_number_read_file]

					if not hydrogen_element:
						# to not redo strip/split every time, save wavelength for the future here
						element_wavelength_dictionary = {}

						# wavelength minimum and maximum for the element (assume sorted)
						wavelength_minimum_element: float = float(lines_file[line_number_read_file].strip().split()[0])
						wavelength_maximum_element: float = float(lines_file[number_of_lines_element+line_number_read_file - 1].strip().split()[0])

						element_wavelength_dictionary[0] = wavelength_minimum_element
						element_wavelength_dictionary[number_of_lines_element - 1] = wavelength_maximum_element

						# check that ANY wavelengths are within the range at all
						if not (wavelength_maximum_element < segment_min_wavelength or wavelength_minimum_element > segment_max_wavelength):
							for seg_begin, seg_end in zip(segment_to_use_begins, segment_to_use_ends):  # wavelength lines write here
								index_seg_start, element_wavelength_dictionary = binary_search_lower_bound(lines_file[line_number_read_file:number_of_lines_element + line_number_read_file],
																										   element_wavelength_dictionary, 0, number_of_lines_element - 1, seg_begin)
								wavelength_current_line: float = element_wavelength_dictionary[index_seg_start]
								line_stripped: str = lines_file[line_number_read_file + index_seg_start].strip()
								line_number_read_for_element: int = index_seg_start + line_number_read_file
								while wavelength_current_line <= seg_end and line_number_read_for_element < number_of_lines_element + line_number_read_file - 1:
									all_lines_to_write.append(f"{line_stripped} \n")
									line_number_read_for_element += 1
									line_stripped: str = lines_file[line_number_read_for_element].strip()
									wavelength_current_line: float = float(line_stripped.split()[0])
					else:
						while line_number_read_for_element < number_of_lines_element:
							line_stripped: str = lines_file[line_number_read_for_element + line_number_read_file].strip()
							all_lines_to_write.append(f"{line_stripped} \n")
							line_number_read_for_element += 1

					line_number_read_file: int = number_of_lines_element + line_number_read_file

				if len(all_lines_to_write) > 0:
					write_lines(all_lines_to_write, elem_line_1_to_save, elem_line_2_to_save, new_file_to_write)
					file_not_empty: bool = True
				if not file_not_empty:
					os.system(f"rm {new_linelist}")

def binary_search_lower_bound(array_to_search: list[str], dict_array_values: dict, low: int, high: int, element_to_search: float) -> tuple[int, dict]:
	"""
	Gives out the lower index where the value is located between the ranges. For example, given array [12, 20, 32, 40, 52]
	Value search: 5, result: 0
	Value search: 13, result: 1
	Value search: 20, result: 1
	Value search: 21, result: 2
	Value search: 51 or 52 or 53, result: 4
	:param array_to_search:
	:param dict_array_values:
	:param low:
	:param high:
	:param element_to_search:
	:return:
	"""
	while low < high:
		middle: int = low + (high - low) // 2

		if middle not in dict_array_values:
			dict_array_values[middle] = float(array_to_search[middle].strip().split()[0])
		array_element_value: float = dict_array_values[middle]

		if array_element_value < element_to_search:
			low: int = middle + 1
		else:
			high: int = middle
	return low, dict_array_values

def write_lines(all_lines_to_write: list[str], elem_line_1_to_save: str, elem_line_2_to_save: str, new_file_to_write: typing.TextIO):
	new_file_to_write.write(f"{elem_line_1_to_save}	{len(all_lines_to_write)}\n")
	new_file_to_write.write(f"{elem_line_2_to_save}")
	for line_to_write in all_lines_to_write:
		#pass
		new_file_to_write.write(line_to_write)
