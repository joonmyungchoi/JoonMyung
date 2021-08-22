def read(file_path):
    import json
    import os
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            anno_file = json.load(f)
        return anno_file
    else:
        return False
