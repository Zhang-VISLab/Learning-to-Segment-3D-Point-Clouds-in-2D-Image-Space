import h5py
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
current_palette = sns.color_palette('bright',10)
#%% initialization
result_path = 'ShapeNet_testing_result.hdf5'
class_name = np.genfromtxt('hdf5_data/all_object_categories.txt',dtype='U')[:,0]
#############################################################################
# ['Airplane' 'Bag' 'Cap' 'Car' 'Chair' 'Earphone' 'Guitar' 'Knife'         #
# 'Lamp' 'Laptop' 'Motorbike' 'Mug' 'Pistol' 'Rocket' 'Skateboard' 'Table'] #
#############################################################################

#%% load the test set
print("loading testing data")
f = h5py.File(result_path,'r')
x_test = f['x_test']
y_test = f['y_test'][:]
s_test = f['s_test']
p_test = f['p_test']
pre_test = f['pre_test']

#%% select a test sample
idx_class = 0
idx_class_sample = 0

idx_sample_list = np.where(y_test==idx_class)[0]
idx_sample = idx_sample_list[idx_class_sample]
label_min = s_test[idx_sample_list].min()
label_max = s_test[idx_sample_list].max()
print('Class_name:',class_name[idx_class],', test sample id:',idx_sample)
#%% load the test sample
x_pt = x_test[idx_sample]
s_pt = s_test[idx_sample]-label_min
pre_pt = pre_test[idx_sample]-label_min


#%% visulize the reconstructed points
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

for i_seg in range(label_max - label_min +1):
    idxs = np.where(s_pt == i_seg)[0]
    color = current_palette.as_hex()[i_seg]
    ax.scatter(x_pt[idxs,0], x_pt[idxs,1], x_pt[idxs,2],marker='.',c=color,s=2,label='Category '+str(i_seg))
ax.set_title('Ground Truth')
ax.legend()
plt.show()


#%% visulize the reconstructed points
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

for i_seg in range(label_max - label_min +1):
    idxs = np.where(pre_pt == i_seg)[0]
    color = current_palette.as_hex()[i_seg]
    ax.scatter(x_pt[idxs,0], x_pt[idxs,1], x_pt[idxs,2],marker='.',c=color,s=2,label='Category '+str(i_seg))
ax.set_title('Network Output')
ax.legend()
plt.show()