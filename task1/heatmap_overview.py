import matplotlib.pyplot as plt
import numpy as np

# plots a heatmap of the following:
# - configuartion boundary (1024 hash functions)
# - precision
# - recall
# - F-measure

A = np.load("A.npy")

# create a map that shows the precision of all configurations which
# have a recall of almost 1 and
B = np.zeros((np.shape(A)[1],np.shape(A)[1]))
for i in range(0, np.shape(A)[1]):
    for j in range(0, np.shape(A)[1]):
        # if the recall is very good!
        if A[1, i, j] > 0.9999:
            # save the precision (if it's good enough) that we can get in B
            if A[0,i,j] > 0.38:
                B[i,j] = A[0,i,j]
            else:
                # just add a small value to indicate that the recall would be 1
                # there.
                B[i,j] = -A[0,i,j]

# create plot
f = plt.figure()

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='all', sharey='all')

ax1.imshow(B, cmap='hot', interpolation='nearest')
ax1.set_title('Prec. (>0.38) for Conf. with Recall ~ 1.0')

ax2.imshow(A[0,:,:], cmap='hot', interpolation='nearest')
ax2.set_title('Precision')

ax3.imshow(A[1,:,:], cmap='hot', interpolation='nearest')
ax3.set_title('Recall')

ax4.imshow(A[2,:,:], cmap='hot', interpolation='nearest')
ax4.set_title('F-Measure')


ax1.set_xlabel('r (rows per band)')
ax1.set_ylabel('b (bands)')
ax2.set_xlabel('r (rows per band)')
ax2.set_ylabel('b (bands)')
ax3.set_xlabel('r (rows per band)')
ax3.set_ylabel('b (bands)')
ax4.set_xlabel('r (rows per band)')
ax4.set_ylabel('b (bands)')

plt.gca().invert_yaxis()
plt.show()

