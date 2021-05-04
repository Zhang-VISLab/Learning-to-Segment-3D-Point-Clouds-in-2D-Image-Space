import os
import h5py
import numpy as np
BASE_DIR= 'hdf5_data/'

#%% analyze the dataset
print("---------------")
print("analyze dataset")
print("---------------")
dir_list = os.listdir(BASE_DIR)

train_files = [f for f in dir_list if f.startswith("ply_data_train")]
val_files =  [f for f in dir_list if f.startswith("ply_data_val")]
test_files = [f for f in dir_list if f.startswith("ply_data_test")]
train_files = np.sort(train_files)
val_files   = np.sort(val_files)
test_files = np.sort(test_files)

train_length = 0
for train_file in train_files:
    with h5py.File(BASE_DIR+train_file,'r') as f:
        train_length += len(f['label'])

val_length = 0
for val_file in val_files:
    with h5py.File(BASE_DIR+val_file,'r') as f:
        val_length += len(f['label'])
        
test_length = 0
for test_file in test_files:
    with h5py.File(BASE_DIR+test_file,'r') as f:
        test_length += len(f['label'])

print('training labels:',train_length,'testing labels:',test_length)
#%% combine dataset
print("---------------")
print("combine dataset")
print("---------------")
f0 = h5py.File('ShapeNet_training.hdf5','w')

x_train =f0.create_dataset("x_train", (train_length,2048,3), dtype='<f4')
y_train =f0.create_dataset("y_train", (train_length,1), dtype='|u1')
s_train =f0.create_dataset("s_train", (train_length,2048), dtype='|u1')

x_val =f0.create_dataset("x_val", (val_length,2048,3), dtype='<f4')
y_val =f0.create_dataset("y_val", (val_length,1), dtype='|u1')
s_val =f0.create_dataset("s_val", (val_length,2048), dtype='|u1')

x_test =f0.create_dataset("x_test", (test_length,2048,3), dtype='<f4')
y_test =f0.create_dataset("y_test", (test_length,1), dtype='|u1')
s_test =f0.create_dataset("s_test", (test_length,2048), dtype='|u1')

offset = 0
for train_file in train_files:
    with h5py.File(BASE_DIR+train_file,'r') as f:
        print("combining dataset:", train_file)
        dataset_length = len(f['label'])
        x_train[offset:offset+dataset_length] =f['data']
        y_train[offset:offset+dataset_length] =f['label']
        s_train[offset:offset+dataset_length] =f['pid']
        offset += dataset_length

offset = 0
for val_file in val_files:
    with h5py.File(BASE_DIR+val_file,'r') as f:
        print("combining dataset:", val_file)
        dataset_length = len(f['label'])
        x_val[offset:offset+dataset_length] =f['data']
        y_val[offset:offset+dataset_length] =f['label']
        s_val[offset:offset+dataset_length] =f['pid']
        offset += dataset_length
        
offset = 0
for test_file in test_files:
    with h5py.File(BASE_DIR+test_file,'r') as f:
        print("combining dataset:", test_file)
        dataset_length = len(f['label'])
        x_test[offset:offset+dataset_length] =f['data']
        y_test[offset:offset+dataset_length] =f['label']
        s_test[offset:offset+dataset_length] =f['pid']
        offset += dataset_length

#%%
f0.close()
