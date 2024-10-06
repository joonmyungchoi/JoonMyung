import math
from math import exp
# Floating Point
# 1(부호) | 11(지수) | 52(가수) → 64개

# Question A.
print((0.7 - 0.4) == 0.3) # False

# Question B.
print(math.pow(2, 66) + math.pow(2, 13) + 1 == math.pow(2, 66))   # True
print(math.pow(2, 66) + (math.pow(2, 13) + 1) == math.pow(2, 66)) # False

# Question C.
print(exp(1000) / (1 + exp(1000)))