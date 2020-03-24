#%% load librarys
# load support python files in local path
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import time

import h5py
import numpy as np


from fun_GPGL_partnet import GPGL2_seg
#%% Global Settings
NUM_REPEATS = 15
SIZE_SUB = 16
SIZE_TOP = 16
NUM_CUTS = 100
NUM_POINTS = 10000
FLAG_ROTATION = 0
FLAG_JITTER   = 1

#%% category ID to prepare dataset
# Since there are too many shapes in PartNet dataset, it is
# recommended to prepare them one by one in case any shutdown.

category_index = 0

#%% category list
dataset_dir = os.path.join(BASE_DIR,'sem_seg_h5')
dir_list = ['Knife-1',
 'Faucet-1',
 'Door-1',
 'Bowl-1',
 'Lamp-1',
 'Refrigerator-2',
 'Bag-1',
 'Bed-1',
 'Table-3',
 'Keyboard-1',
 'Lamp-3',
 'Bottle-1',
 'Chair-3',
 'Refrigerator-3',
 'Laptop-1',
 'Door-2',
 'Clock-1',
 'Chair-2',
 'Bottle-3',
 'Earphone-3',
 'Bed-2',
 'Scissors-1',
 'Vase-3',
 'Faucet-3',
 'Dishwasher-3',
 'Display-1',
 'Vase-1',
 'Earphone-1',
 'Display-3',
 'Hat-1',
 'StorageFurniture-3',
 'Table-1',
 'Microwave-1',
 'Dishwasher-1',
 'Lamp-2',
 'Clock-3',
 'Microwave-3',
 'StorageFurniture-1',
 'Dishwasher-2',
 'Bed-3',
 'Mug-1',
 'TrashCan-1',
 'Table-2',
 'Refrigerator-1',
 'Door-3',
 'Knife-3',
 'Chair-1',
 'StorageFurniture-2',
 'Microwave-2',
 'TrashCan-3']

dir_list = dir_list[category_index:category_index+1]

#%% analyze the dataset

print("---------------")
print("Analyzing the PartNet dataset ")
print("---------------")
train_dataset_length = 0
train_label_length = 0

test_dataset_length = 0
test_label_length = 0

label_offset = []
label_offset.append(0)
for idx, category in zip(range(len(dir_list)),dir_list):
    print("Analyzing", idx,category)
    category_dir = os.path.join(dataset_dir,category)
    train_file_path = os.path.join(category_dir,'train_files.txt')
    val_file_path = os.path.join(category_dir,'val_files.txt')
    test_file_path = os.path.join(category_dir,'test_files.txt')

    with open(train_file_path, 'r') as fin:
        train_file = fin.readlines()
        train_files = [f[2:-1] for f in train_file]

    f_length = 0
    max_label = 0
    for file in train_files:
        with h5py.File(os.path.join(category_dir,file),'r') as f:
            f_length += len(f['label_seg'])
            max_label = np.maximum(np.array(f['label_seg']).max()+1,max_label)

    train_dataset_length += f_length
    train_label_length += max_label

    f_length = 0
    max_label = 0
    with open(test_file_path, 'r') as fin:
        test_file = fin.readlines()
        test_files = [f[2:-1] for f in test_file]

    for file in test_files:
        with h5py.File(os.path.join(category_dir,file),'r') as f:
            f_length += len(f['label_seg'])
            max_label = np.maximum(np.array(f['label_seg']).max()+1,max_label)

    test_dataset_length += f_length
    test_label_length += max_label

    label_offset.append(max_label)


assert(test_label_length==train_label_length)
data_info = {"train_dataset_length":train_dataset_length,
                 "test_dataset_length":test_dataset_length,
                 "label_length":test_label_length}


print("---------------")
print("Total training samples:",train_dataset_length)
print("Total test samples:",test_dataset_length)
print("Total part categories:",test_label_length)
print("---------------")

#%%
def parepare_dataset(sess,num_repeats):
    data_file_size = data_info[sess+"_dataset_length"]
    NUM_CLASSES = data_info["label_length"]
    x_set = f0.create_dataset("x_"+sess, (data_file_size*num_repeats,SIZE_SUB*SIZE_TOP,SIZE_SUB*SIZE_TOP,3), dtype='f')
    s_set = f0.create_dataset("s_"+sess, (data_file_size*num_repeats,SIZE_SUB*SIZE_TOP,SIZE_SUB*SIZE_TOP,NUM_CLASSES+1), dtype='f')
    y_set = f0.create_dataset("y_"+sess, (data_file_size*num_repeats,NUM_POINTS,3), dtype='f')
    p_set = f0.create_dataset("p_"+sess, (data_file_size*num_repeats,NUM_POINTS,2), dtype='i')
    l_set = f0.create_dataset("l_"+sess, (data_file_size*num_repeats,NUM_POINTS), dtype='i')
    d_set = f0.create_dataset("d_"+sess, (data_file_size*num_repeats,NUM_POINTS,3), dtype='i')
    c_set = f0.create_dataset("c_"+sess, (data_file_size*num_repeats,20), dtype='S10')


    sample_list = np.arange(data_file_size*num_repeats)
    if(sess=='train'):
        np.random.shuffle(sample_list)

    idx_sample = 0
    node_loss_rate_list = []
    time_begin = time.time_ns()

    for category_idx, category in zip(range(len(dir_list)),dir_list):
        category_dir = os.path.join(dataset_dir,category)
        file_path = os.path.join(category_dir,sess+'_files.txt')



        with open(file_path, 'r') as fin:
            files = fin.readlines()
            files = [f[2:-1] for f in files]

        for file in files:
            with h5py.File(os.path.join(category_dir,file),'r') as f:
                file_data, file_label, file_seg = f['data'][:], category_idx, f['label_seg'][:]
                file_seg += label_offset[category_idx]
                for data, label in zip(file_data,file_seg):
                    for i_repeats in (range(num_repeats)):
                        mat, seg, pos, node_loss_rate = GPGL2_seg(data,label,NUM_CLASSES, NUM_CUTS,SIZE_TOP,SIZE_SUB)

                        x_set[sample_list[idx_sample]] = mat
                        y_set[sample_list[idx_sample],category_idx] = 1
                        s_set[sample_list[idx_sample]] = seg
                        p_set[sample_list[idx_sample]] = pos
                        l_set[sample_list[idx_sample]] = label
                        d_set[sample_list[idx_sample]] = data
                        c_set[sample_list[idx_sample]] = np.string_(dir_list[category_idx])

                        print(dataset_name+":"+sess+": idx="+str(idx_sample)+"/"+str(x_set.shape[0]),"node_loss_rate="+str(node_loss_rate))
                        idx_sample+=1
                        if(idx_sample==data_file_size*num_repeats):
                            break
                        node_loss_rate_list.append(node_loss_rate)
        time_end = time.time_ns()
        node_loss_rate_final =np.array(node_loss_rate_list).mean()
        x_set.attrs['NUM_REPEATS']=num_repeats
        x_set.attrs['node loss ratio']=node_loss_rate_final
    return idx_sample,node_loss_rate_final,time_end-time_begin

#%% building dataset

dataset_name = "PartNet_"+str(category_index)+"_"+dir_list[0]+"_"+str(NUM_REPEATS)+"x"
if(FLAG_ROTATION):
    dataset_name += '_r'
if(FLAG_JITTER):
    dataset_name += '_j'
f0 = h5py.File(dataset_name+'.hdf5', 'w')

train_sample,train_node_loss,train_time = parepare_dataset('train',NUM_REPEATS)
test_sample,test_node_loss,test_time = parepare_dataset('test',1)
f0.close()
print("train_sample:",train_sample,"train_node_loss:",train_node_loss,"train_time:",train_time/train_sample/1e6,"ms/sample")
print("test_sample:",test_sample,"test_node_loss:",test_node_loss,"test_time:",test_time/test_sample/1e6,"ms/sample")








