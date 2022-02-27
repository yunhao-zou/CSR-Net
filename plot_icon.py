import matplotlib.pyplot as plt
import numpy as np
import os
for i in range(14):
    cmap = np.zeros((8, 8), dtype=np.int32)
    cmap[:,:] = i
    w, h = cmap.shape
    plt.figure(figsize=[h/100.0, w/100.0])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    plt.axis('off')
    plt.imshow(cmap, cmap='jet', vmin=0, vmax=13)
    plt.savefig(os.path.join('colormap/KSC/' + str(i) + '.png'), format='png')
    plt.close()