import numpy as np
import subprocess
import ast

# script to perform grid-search for the hashing parameters

# parameters (update them for further grid searches)
# Y
MIN_NUM_BANDS = 26
MAX_NUM_BANDS = 40

# X
MIN_NUM_ROWS_PER_BAND = 21
MAX_NUM_ROWS_PER_BAND = 40

# try to load matrix A or initialize it
A = np.zeros((3, 50, 50))
try:
    A = np.load("A.npy")
except:
    for i in range(1, 50):
        for j in range(1, 50):
            if i*j > 1024:
                A[:,i,j] = -0.3

for i in range(MIN_NUM_BANDS, MAX_NUM_BANDS+1):
    for j in range(MIN_NUM_ROWS_PER_BAND, MAX_NUM_ROWS_PER_BAND+1):
        if i*j<=1024:

            print (i,j)

            with open('mapreduce.template', 'r') as file:
                filedata = file.read()

            # Replace the target string
            filedata = filedata.replace('{{BANDS}}', str(i))
            filedata = filedata.replace('{{ROWS_PER_BAND}}', str(j))

            # Write the file out again
            with open('mapreduce.py', 'w') as file:
                file.write(filedata)

            proc = subprocess.Popen(['python', 'runner2.py', 'mapreduce.py'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)

            # Parse result
            s = proc.communicate()[0]
            print(s)
            t = ast.literal_eval(s)

            A[0, i, j] = t[0]
            A[1, i, j] = t[1]
            A[2, i, j] = t[2]

            print A[:,i,j]
            np.save("A.npy", A)