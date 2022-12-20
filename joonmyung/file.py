
import json
import os

def read(file_path):
    filetype = file_path.split(".")[-1]
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            if filetype == "json":
                return json.load(f)
            elif filetype =="mat":
                from scipy import io
                return io.loadmat(file_path)
            elif filetype == "xml":
                pass
            elif filetype == "pickle":
                pass
    else:
        return False