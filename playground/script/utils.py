from joonmyung.utils import str2list, str2bool
import argparse

def getCompressionParser():
    parser = argparse.ArgumentParser(description="tuning")
    parser.add_argument('--output_dir', default="/hub_data1/joonmyung/project/Compression", type=str)
    parser.add_argument('--result_name', default="1.0.0", type=str)
    parser.add_argument('--batch_size', default='1', type=int)

    parser.add_argument('--compression', default=[[1, 0, 10, 0, 1, 1], [1, 10, 25, 1], [0]], type=str2list)
    parser.add_argument('--r_merge', default=None, type=str2list)
    args = parser.parse_args()
    return args