import csv

n = 0
lis = []
for i in range(150):
    lis.append(str(n))
    n = n + 10

write_file = "time1.csv"
with open(write_file, "w") as output:
    for line in lis:
        output.write(line + '\n')
#write_csv('time1.csv', lis)