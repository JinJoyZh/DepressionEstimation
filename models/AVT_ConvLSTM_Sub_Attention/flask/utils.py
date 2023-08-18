import os
import re


def get_sorted_files(path, suffix):
    file_names = os.listdir(path)
    csv_files = []
    for file_name in file_names:
        if(file_name.endswith(suffix)):
            csv_files.append(file_name)
    def get_key(elem):
        try:
            index = int(re.findall(r"\d+",elem)[-1])
            return index
        except ValueError:
            return -1
    csv_files.sort(key = get_key)
    return csv_files