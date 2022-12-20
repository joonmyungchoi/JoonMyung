# 1. Yappi
def add(a, b):
    return a+b


def get_sum_of_list():
    final_list = []
    for i in range(10000):
        out = add(i, i)
        final_list.append(out)
    return final_list

if __name__ == "__main__":
    l = get_sum_of_list()
    print(l)


# 2. cProfile