def read(file_path):
    import json
    import os

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

    else:
        return False



# read("../sample/caption.json")
read("../sample/data.mat")