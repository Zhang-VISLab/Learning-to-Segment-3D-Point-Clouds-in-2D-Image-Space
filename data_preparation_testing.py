# import numpy as np
import json
import os
BASE_DIR = './'
import provider


# load common librarys
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import time
import h5py

# load GPGL functions
from fun_GPGL import graph_cut, fun_graph_cosntruct,fun_GPGL_layout_push

#%% global settings
NUM_POINTS = 2048
NUM_REPEATS = 1
NUM_CLASSES = 40
NUM_CUTS = 32
SIZE_SUB = 16
SIZE_TOP = 16
NUM_CUTPOINTS = int(NUM_POINTS/NUM_CUTS)

FLAG_ROTATION = 0
FLAG_JITTER   = 0

#%%
kmeans_solver = KMeans(n_clusters=NUM_CUTS, n_init=1,max_iter=100)
#%%
def GPGL2_seg(data,current_sample_seg):
    data = data+np.random.rand(len(data),len(data[0]))*1e-6
    dist_mat = kmeans_solver.fit_transform(data)


    node_top,labels = graph_cut(data,dist_mat,NUM_POINTS,NUM_CUTS)

    aij_mat = fun_graph_cosntruct(node_top)

    H = nx.from_numpy_matrix(aij_mat)
    pos_spring = nx.spring_layout(H)
    pos_spring = np.array([pos for idx,pos in sorted(pos_spring.items())])

    pos = fun_GPGL_layout_push(pos_spring,SIZE_SUB)
    pos_top = fun_GPGL_layout_push(pos_spring,SIZE_TOP)

    ##%%
    pos_cuts = []
    for i_cut in range(NUM_CUTS):
        pos_cut_3D = data[labels==i_cut,:]

        if(len(pos_cut_3D)<5):
            pos_raw = [[0,0],[0,1],[1,1],[1,0]]
            pos = pos_raw[:len(pos_cut_3D)]
            pos_cuts.append(pos)
            continue
        aij_mat = fun_graph_cosntruct(pos_cut_3D)
        H = nx.from_numpy_matrix(aij_mat)
        pos_spring = nx.spring_layout(H)
        pos_spring = np.array([pos for idx,pos in sorted(pos_spring.items())])
        pos = fun_GPGL_layout_push(pos_spring,SIZE_SUB)


        pos_cuts.append(pos)


    ##%% combine all layout positions
    cuts_count = np.zeros(NUM_CUTS).astype(np.int)
    pos_all = []
    for idx in range(NUM_POINTS):
        label = labels[idx]
        pos_all.append(pos_cuts[label][cuts_count[label]]+pos_top[label]*SIZE_SUB)
        cuts_count[label] +=1
    pos_all=np.array(pos_all)

    ##%% assign all features into the grid map
    mat = np.zeros([SIZE_SUB*SIZE_TOP,SIZE_SUB*SIZE_TOP,3])
    seg = np.zeros([SIZE_SUB*SIZE_TOP,SIZE_SUB*SIZE_TOP,51])
    seg[:,:,-1] = 1

    for data1,seg1, pos in zip(data,current_sample_seg,pos_all):
        mat[pos[0],pos[1]]=data1
        seg[pos[0],pos[1],int(seg1)]=1
        seg[pos[0],pos[1],-1]=0

    num_nodes_m = np.sum(np.linalg.norm(mat,axis=-1)>0)
    node_loss_rate=(1- num_nodes_m/NUM_POINTS)
    return mat, seg, pos_all, node_loss_rate




#%%
hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data/')

color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
color_map = json.load(open(color_map_file, 'r'))

all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
fin = open(all_obj_cats_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
fin.close()

all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))
NUM_CATEGORIES = 16
NUM_PART_CATS = len(all_cats)

TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
TESTING_FILE_LIST = os.path.join(hdf5_data_dir, 'val_hdf5_file_list.txt')
#%%
train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
num_train_file = len(train_file_list)
test_file_list = provider.getDataFiles(TESTING_FILE_LIST)
num_test_file = len(test_file_list)

#%%
def parepare_dataset(sess,NUM_REPEATS,file_name):
    data_files =provider.getDataFiles( \
        os.path.join(hdf5_data_dir, sess+'_hdf5_file_list.txt'))

    data_file_size = 0
    for fn in range(len(data_files)):
        cur_train_filename = os.path.join(hdf5_data_dir, data_files[fn])
        current_data, current_label, current_seg= provider.loadDataFile_with_seg(cur_train_filename)
        file_size = current_data.shape[0]
        data_file_size +=file_size

    x_set = f.create_dataset("x_"+sess, (data_file_size*NUM_REPEATS,SIZE_SUB*SIZE_TOP,SIZE_SUB*SIZE_TOP,3), dtype='f')
    y_set = f.create_dataset("y_"+sess, (data_file_size*NUM_REPEATS,16), dtype='i')
    s_set = f.create_dataset("s_"+sess, (data_file_size*NUM_REPEATS,SIZE_SUB*SIZE_TOP,SIZE_SUB*SIZE_TOP,51), dtype='i')
    p_set = f.create_dataset("p_"+sess, (data_file_size*NUM_REPEATS,NUM_POINTS,2), dtype='i')
    l_set = f.create_dataset("l_"+sess, (data_file_size*NUM_REPEATS,NUM_POINTS), dtype='i')
    d_set = f.create_dataset("d_"+sess, (data_file_size*NUM_REPEATS,NUM_POINTS,3), dtype='f')


    sample_list = np.arange(data_file_size*NUM_REPEATS)
    if(sess=='train'):
        np.random.shuffle(sample_list)

    idx_sample = 0
    node_loss_rate_list = []
    time_begin = time.time()
    for fn in range(len(data_files)):
        cur_train_filename = os.path.join(hdf5_data_dir, data_files[fn])
        file_data, file_label, file_seg= provider.loadDataFile_with_seg(cur_train_filename)
        current_data = file_data[:,0:NUM_POINTS,:]
        current_seg  = file_seg[:,0:NUM_POINTS]

        for current_sample_data,current_sample_label,current_sample_seg in zip(current_data,file_label,current_seg):
            for i_repeat in range(NUM_REPEATS):
                final_data = current_sample_data[np.newaxis,:,:]
                if(FLAG_ROTATION):
                    final_data = provider.rotate_point_cloud(final_data)
                if(FLAG_JITTER):
                    final_data = provider.jitter_point_cloud(final_data)

                mat, seg, pos, node_loss_rate = GPGL2_seg(final_data[0],current_sample_seg)

                x_set[sample_list[idx_sample]] = mat
                y_set[sample_list[idx_sample],current_sample_label] = 1
                s_set[sample_list[idx_sample]] = seg
                p_set[sample_list[idx_sample]] = pos
                l_set[sample_list[idx_sample]] = current_sample_seg
                d_set[sample_list[idx_sample]] = current_sample_data

                print(file_name+":"+sess+": idx="+str(idx_sample)+"/"+str(x_set.shape[0]),"node_loss_rate="+str(node_loss_rate))
                idx_sample+=1
                node_loss_rate_list.append(node_loss_rate)
    time_end = time.time()
    node_loss_rate_final =np.array(node_loss_rate_list).mean()
    x_set.attrs['NUM_REPEATS']=NUM_REPEATS
    x_set.attrs['node loss ratio']=node_loss_rate_final
    return idx_sample,node_loss_rate_final,time_end-time_begin


#%% main call function
file_name = 'ShapeNet_testing.hdf5'

f = h5py.File(file_name, 'w')
test_sample,test_node_loss,test_time = parepare_dataset('test',1,file_name)
f.close()
print("test_sample:",test_sample,"test_node_loss:",test_node_loss,"test_time:",test_time/test_sample/1e-3,"ms/sample")





