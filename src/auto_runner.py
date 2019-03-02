import subprocess
import sys

def check_end(line):
    print(line, end='')
    if line == '#!#!#! shutdown':
        print('go shutdown!')

if __name__ == '__main__':
    proc = subprocess.Popen(['python'] + sys.argv[1:], stdout=subprocess.PIPE)
    for line in iter(proc.stdout):
        check_end(line.decode('utf-8'))
