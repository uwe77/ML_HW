from gwo import GWO
from data import *
import re


d_space = data_space()
f = open('iris.txt', 'r')
for line in f.readlines():
    s = re.split(r'\s+', line)
    data_input = data(int(s[4]))
    data_input[:] = [float(i) for i in s[:4]]
    d_space.append_data(data_input)
f.close()

gwo = GWO()
gwo.run(d_space)