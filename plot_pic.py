import matplotlib.pyplot as plt
import scipy.io as sio
import os
de_map = sio.loadmat('data/paviaU_gt.mat')
de_map = de_map['paviaU_gt']
print(de_map.shape)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
plt.margins(0,0)
plt.axis('off')
plt.imsave('test.png', de_map)
# plt.pcolor(de_map, cmap='jet')
plt.imshow(de_map, cmap='jet')
plt.savefig(os.path.join('test.png'), format='png')
plt.close()