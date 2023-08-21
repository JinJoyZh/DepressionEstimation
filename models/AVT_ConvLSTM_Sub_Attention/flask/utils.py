import os
import random
import re
import numpy as np

import torch


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

def init_seed(manual_seed):
    """
    Set random seed for torch and numpy.
    """
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False