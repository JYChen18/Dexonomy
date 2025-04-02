import os

os.system("ps aux | grep dexonomy.main > debug.txt")

with open("debug.txt", "r") as f:
    x = f.readlines()

print(x)
flag = input("Press 'c' to kill all the above processes")
if flag != "c":
    exit(1)

for l in x:
    pid = l.split(" ")[1]
    os.system(f"kill -9 {pid}")
