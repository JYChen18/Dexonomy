import os

os.system("ps aux | grep dexonomy.main > debug.txt")

with open("debug.txt", "r") as f:
    x = f.readlines()

print(x)
flag = input("Press 'c' to kill all the above processes")
if flag != "c":
    exit(1)

for line in x:
    parts = [part for part in line.split(" ") if part]
    pid = parts[1]
    print(f"kill -9 {pid}")
    os.system(f"kill -9 {pid}")
