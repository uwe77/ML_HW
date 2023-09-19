import re
y = []
x = []
f = open('iris.txt', 'r')
for line in f.readlines():
    s = re.split(r'\s+', line)
    temp_list = []
    for i in range(4):
        temp_list.append(float(s[i]))
    y.append(int(s[4]))
    x.append(temp_list)
for i in y:
    print(i)