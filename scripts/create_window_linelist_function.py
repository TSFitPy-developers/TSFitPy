from __future__ import annotations
import os
import numpy as np
import time
import shutil
global time_spent_reading
global time_spent_writing

#@profile
def create_window_linelist(seg_begins: np.ndarray[float], seg_ends: np.ndarray[float], old_path_name: str,
                           new_path_name: str, molecules_flag: bool, lbl=False, do_hydrogen=True):
    global time_spent_reading
    line_list_path: str = old_path_name
    line_list_files: list = [entry.path for entry in os.scandir(line_list_path) if entry.is_file()]

    segment_to_use_begins: np.ndarray = np.asarray(seg_begins)
    segment_to_use_ends: np.ndarray = np.asarray(seg_ends)

    segment_index_order: np.ndarray = np.argsort(segment_to_use_begins)
    segment_to_use_begins: np.ndarray = segment_to_use_begins[segment_index_order]
    segment_to_use_ends: np.ndarray = segment_to_use_ends[segment_index_order]
    segment_min_wavelength: float = np.min(segment_to_use_begins)
    segment_max_wavelength: float = np.max(segment_to_use_ends)

    if not lbl:
        if not os.path.exists(new_path_name):
            os.makedirs(os.path.join(f"{new_path_name}", "0", ''))
    else:
        for i in range(len(seg_begins)):
            new_path_name_one_seg: str = os.path.join(f"{new_path_name}", f"{i}", '')
            if not os.path.exists(new_path_name_one_seg):
                os.makedirs(new_path_name_one_seg)

    for line_list_number, line_list_file in enumerate(line_list_files):
        new_linelist_name: str = f"{new_path_name}"  # f"linelist-{line_list_number}.bsyn"
        with open(line_list_file) as fp:
            # so that we dont read full file if we are not sure that we use it (if it is a molecule)
            time_start1 = time.perf_counter()
            first_line: str = fp.readline()
            time_end1 = time.perf_counter()
            time_spent_reading += time_end1 - time_start1
            fields = first_line.strip().split()
            sep = '.'
            element = fields[0] + fields[1]
            elements = element.split(sep, 1)[0]
            # opens each file, reads first row, if it is long enough then it is molecule. If fitting molecules, then
            # keep it, otherwise ignore molecules
            all_lines_to_write: dict = {}
            if len(elements) > 3 and molecules_flag or len(elements) <= 3:
                if elements == '01.000000' and do_hydrogen:
                    # instead we just copy the file
                    # use shutil.copyfile instead of open and write
                    if not lbl:
                        new_linelist_name: str = os.path.join(f"{new_path_name}", "0",
                                                              f"linelist-{line_list_number}.bsyn")
                        shutil.copyfile(line_list_file, new_linelist_name)
                    else:
                        for seg_index in range(len(seg_begins)):
                            new_linelist_name: str = os.path.join(f"{new_path_name}", f"{seg_index}",
                                                                  f"linelist-{line_list_number}.bsyn")
                            shutil.copyfile(line_list_file, new_linelist_name)

                else:
                    # now read the whole file
                    time_start1 = time.perf_counter()
                    lines_file: list[str] = fp.readlines()
                    time_end1 = time.perf_counter()
                    time_spent_reading += time_end1 - time_start1

                    line_number_read_file: int = 0
                    # append the first line to the lines_file
                    total_lines_in_file: int = len(lines_file) + 1
                    line: str = first_line
                    first_line_read: bool = True
                    while line_number_read_file + 1 < total_lines_in_file:  # go through all line
                        if not first_line_read:
                            line: str = lines_file[line_number_read_file]
                            line_number_read_file += 1
                        else:
                            first_line_read: bool = False
                        fields: list[str] = line.strip().split()

                        if len(fields[0]) > 1:  # save the first two lines of an element for the future
                            elem_line_1_to_save: str = f"{fields[0]} {fields[1]}  {fields[2]}"  # first line of the element
                            number_of_lines_element: int = int(fields[3])
                        else:
                            elem_line_1_to_save: str = f"{fields[0]}   {fields[1]}            {fields[2]}    {fields[3]}"
                            number_of_lines_element: int = int(fields[4])

                        line: str = lines_file[line_number_read_file]
                        elem_line_2_to_save: str = f"{line.strip()}\n"  # second line of the element

                        # now we are reading the element's wavelength and stuff
                        line_number_read_file += 1

                        # to not redo strip/split every time, save wavelength for the future here
                        element_wavelength_dictionary = {}

                        # wavelength minimum and maximum for the element (assume sorted)
                        wavelength_minimum_element: float = float(lines_file[line_number_read_file].strip().split()[0])
                        wavelength_maximum_element: float = float(lines_file[number_of_lines_element + line_number_read_file - 1].strip().split()[0])

                        element_wavelength_dictionary[0] = wavelength_minimum_element
                        element_wavelength_dictionary[number_of_lines_element - 1] = wavelength_maximum_element

                        # check that ANY wavelengths are within the range at all
                        if not (wavelength_maximum_element < segment_min_wavelength or wavelength_minimum_element > segment_max_wavelength):
                            for seg_index, (seg_begin, seg_end) in enumerate(zip(segment_to_use_begins, segment_to_use_ends)):  # wavelength lines write here
                                index_seg_start = binary_find_left_segment_index(lines_file, element_wavelength_dictionary,
                                                                                 0, number_of_lines_element,
                                                                                 line_number_read_file, seg_begin)
                                wavelength_current_line: float = element_wavelength_dictionary[index_seg_start]
                                if seg_begin <= wavelength_current_line <= seg_end:
                                    index_seg_end = binary_find_right_segment_index(lines_file, element_wavelength_dictionary,
                                                                                    index_seg_start, number_of_lines_element,
                                                                                    line_number_read_file, seg_end)

                                    if lbl:
                                        seg_current_index = seg_index
                                    else:
                                        seg_current_index = 0
                                    all_lines_to_write[seg_current_index] = lines_file[index_seg_start + line_number_read_file:index_seg_end + line_number_read_file + 1]

                        line_number_read_file: int = number_of_lines_element + line_number_read_file

                        if all_lines_to_write:
                            write_lines(all_lines_to_write, elem_line_1_to_save, elem_line_2_to_save, new_linelist_name,
                                        line_list_number)
                            all_lines_to_write.clear()

def binary_search_lower_bound(array_to_search: list[str], dict_array_values: dict, low: int, high: int,
                              element_to_search: float) -> int:
    """
	Gives out the upper index where the value is located between the ranges. For example, given array [12, 20, 32, 40, 52]
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
    if element_to_search >= float(array_to_search[-1].strip().split()[0]):
        return min(high, len(array_to_search) - 1)
    while low < high:
        middle: int = low + (high - low) // 2

        if middle not in dict_array_values:
            dict_array_values[middle] = float(array_to_search[middle].strip().split()[0])
        array_element_value: float = dict_array_values[middle]

        if array_element_value < element_to_search:
            low: int = middle + 1
        else:
            high: int = middle
    return low

def binary_find_left_segment_index(array_to_search: list[str], dict_array_values: dict, low: int, high: int, offset_array: int,
                                   element_to_search: float):
    """actually new?  For example, given array [12, 20, 32, 40, 52]
	Value search: 5, result: 0
	Value search: 13, result: 1
	Value search: 20, result: 1
	Value search: 21, result: 2
	Value search: 51 or 52 result: 4
	Value search: 53, result: 5
	"""
    # if value to search is less than the first element, return 0
    if element_to_search < float(array_to_search[low + offset_array].strip().split()[0]):
        return low
    # if value to search is greater than the last element, return high
    if element_to_search > float(array_to_search[high + offset_array - 1].strip().split()[0]):
        return high

    left, right = 0, high

    while left <= right:
        mid = (left + right) // 2
        if mid + low not in dict_array_values:
            dict_array_values[mid + low] = float(array_to_search[mid + low + offset_array].strip().split()[0])
        mid_value: float = dict_array_values[mid + low]

        if mid_value < element_to_search:
            left = mid + 1
        elif mid_value > element_to_search:
            right = mid - 1
        else:
            # If the exact value is found, check if it is the first occurrence.
            while mid > 0:
                if mid - 1 not in dict_array_values:
                    dict_array_values[mid + low - 1] = float(
                        array_to_search[mid - 1 + low + offset_array].strip().split()[0])
                if dict_array_values[mid + low - 1] == element_to_search:
                    mid -= 1
                else:
                    break
            return mid + low
    return left + low

#@profile
def binary_find_right_segment_index(array_to_search: list[str], dict_array_values: dict, low: int, high: int, offset_array: int,
                                   element_to_search: float):
    """ For example, given array [12, 20, 32, 40, 52]
	Value search: 5, result: 0
	Value search: 13, result: 0
	Value search: 20, result: 1
	Value search: 21, result: 1
	Value search: 51 or 52 result: 3
	Value search: 53, result: 4"""
    high -= 1
    if element_to_search <= float(array_to_search[low + offset_array].strip().split()[0]):
        dict_array_values[0] = float(array_to_search[low + offset_array].strip().split()[0])
        return 0
    if element_to_search >= float(array_to_search[high + offset_array].strip().split()[0]):
        dict_array_values[high] = float(array_to_search[high + offset_array].strip().split()[0])
        return high

    left, right = 0, high - low

    while left < right:
        mid = (left + right) // 2
        if mid + low not in dict_array_values:
            dict_array_values[mid + low] = float(array_to_search[mid + low + offset_array].strip().split()[0])
        mid_value: float = dict_array_values[mid + low]

        if mid_value <= element_to_search:
            left = mid + 1
        else:
            right = mid
    # If not found, left will be the insertion point.
    return left + low - 1

#@profile
def write_lines(all_lines_to_write: dict[list[str]], elem_line_1_to_save: str, elem_line_2_to_save: str,
                new_path_name: str, line_list_number: float):
    global time_spent_writing
    for key in all_lines_to_write:
        new_linelist_name: str = os.path.join(f"{new_path_name}", f"{key}", f"linelist-{line_list_number}.bsyn")
        with open(new_linelist_name, "a") as new_file_to_write:
            new_file_to_write.write(f"{elem_line_1_to_save}	{len(all_lines_to_write[key])}\n")
            new_file_to_write.write(f"{elem_line_2_to_save}")
            time_start1 = time.perf_counter()
            lines_to_write = "".join(all_lines_to_write[key])

            new_file_to_write.write(lines_to_write)
            time_end1 = time.perf_counter()
            time_spent_writing += time_end1 - time_start1


if __name__ == '__main__':
    import shutil
    from tqdm import tqdm
    temp_dir = "./test_test_test/"

    test = True

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    if test:
        time_spent_reading = 0
        time_spent_writing = 0
        time_start = time.perf_counter()
        create_window_linelist([4000], [8000],
                               "/Users/storm/docker_common_folder/TSFitPy/input_files/linelists/linelist_for_fitting/",
                               temp_dir, True, False, True)
        time_end = time.perf_counter()
        total_time_spent = time_end - time_start
        print(f"Total time taken: {total_time_spent}")
        print(f"Time spent reading: {time_spent_reading}")
        print(f"Time spent writing: {time_spent_writing}")
        print(f"Percentage of time spent reading: {time_spent_reading / total_time_spent * 100}%")
        print(f"Percentage of time spent writing: {time_spent_writing / total_time_spent * 100}%")
    else:
        for i in tqdm(range(3)):
            create_window_linelist([4850], [4865], "/Users/storm/docker_common_folder/TSFitPy/input_files/linelists/linelist_for_fitting/",
                                   temp_dir, True, False, True)
            # delete the created folder
            shutil.rmtree(temp_dir)
