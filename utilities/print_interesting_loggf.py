import os

def read_element_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    i = 0
    elements_data = []
    while i < len(lines):
        line_parts = lines[i].split()
        if line_parts[0] == "'":
            atomic_num = (line_parts[1])
        else:
            atomic_num = (line_parts[0])
        ionization = int(line_parts[-2])
        num_lines = int(line_parts[-1])

        element_name = lines[i + 1].strip()

        for _ in range(num_lines):
            i += 1
            data_line = lines[i + 1]
            wavelength, loggf = float(data_line.split()[0]), float(data_line.split()[2])
            elements_data.append((element_name, atomic_num, ionization, wavelength, loggf))

        i += 2

    return elements_data

def find_elements(elements_data, left_wavelength, right_wavelength, loggf_threshold):
    filtered_elements = []
    for element_data in elements_data:
        element_name, atomic_num, ionization, wavelength, loggf = element_data
        if left_wavelength <= wavelength <= right_wavelength and loggf > loggf_threshold:
            filtered_elements.append(element_data)

    sorted_elements = sorted(filtered_elements, key=lambda x: x[3])  # Sort by wavelength

    for element_data in sorted_elements:
        element_name, atomic_num, ionization, wavelength, loggf = element_data
        print(element_name.replace("'", "").replace("NLTE", "").replace("LTE", ""), atomic_num, wavelength, loggf)


if __name__ == '__main__':
    linelist_path = "../input_files/linelists/linelist_for_fitting/"
    linelist_filename = "nlte_ges_linelist_jmg17feb2022_I_II"
    print("element atomic_number  wavelength  loggf")

    left_wavelength = 6645.0 - 0.1  # change this to change the range of wavelengths to print
    right_wavelength = 6645.104
    loggf_threshold = -4            # change this to change the threshold for loggf

    elements_data = read_element_data(os.path.join(linelist_path, linelist_filename))
    find_elements(elements_data, left_wavelength, right_wavelength, loggf_threshold)