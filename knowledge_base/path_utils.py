def is_l3_path(path):
    if 'infos' in path:
        return len(path.split("_")) > 3
    else:
        return len(path.split("_")) > 4

def is_l2_path(path):
    ## infos paths are only l1 and l3
    if 'infos' in path:
        return False
    else:
        return len(path.split("_")) == 4
    
def is_l1_path(path):
    ## infos paths are only l1 and l3
    if 'infos' in path:
        return False
    else:
        return len(path.split("_")) == 3