# import numpy as np
import fun_provider as provider

# load common librarys
import numpy as np
import networkx as nx
from scipy.spatial.distance  import pdist,squareform
from sklearn.cluster import KMeans
import time
import h5py

# load GPGL functions
from fun_GPGL import graph_cut,fun_GPGL_layout_push

#%% global settings
NUM_POINTS = 2048
NUM_REPEATS = 5
NUM_CUTS = 32
SIZE_SUB = 16
SIZE_TOP = 16
NUM_CUTPOINTS = int(NUM_POINTS/NUM_CUTS)

FLAG_ROTATION = 0
FLAG_JITTER   = 1

wall_clock_start = time.time()
#%%
kmeans_solver = KMeans(n_clusters=NUM_CUTS, n_init=1,max_iter=100)
#%% 3D to 2D projection function
def GPGL2_seg(data,current_sample_seg):
    data = data+np.random.rand(len(data),len(data[0]))*1e-6
    dist_mat = kmeans_solver.fit_transform(data)

    node_top,labels = graph_cut(data,dist_mat,NUM_POINTS,NUM_CUTS)

    aij_mat = squareform(pdist(node_top),checks=False)
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

        aij_mat = squareform(pdist(pos_cut_3D),checks=False)
        H = nx.from_numpy_matrix(aij_mat)
        pos_spring = nx.spring_layout(H)
        pos_spring = np.array([pos for idx,pos in sorted(pos_spring.items())])
        pos = fun_GPGL_layout_push(pos_spring,SIZE_SUB)

        pos_cuts.append(pos)

    ##%% combine all layout positions
    cuts_count = np.zeros(NUM_CUTS).astype(np.int64)
    pos_all = []
    for idx in range(NUM_POINTS):
        label = labels[idx]
        pos_all.append(pos_cuts[label][cuts_count[label]]+pos_top[label]*SIZE_SUB)
        cuts_count[label] +=1
    pos_all=np.array(pos_all)

    num_nodes_m = len(np.unique(pos_all,axis=0))
    node_loss_rate=(1- num_nodes_m/NUM_POINTS)
    return pos_all, node_loss_rate


#%%
def parepare_dataset(sess,NUM_REPEATS,file_name):
    f0 = h5py.File('ShapeNet_training.hdf5','r')
    
    data_file_size = len(f0['x_'+sess])

    y_set_out = f.create_dataset("y_"+sess, (data_file_size*NUM_REPEATS,1), dtype='i')                                     # point cloud category
    p_set_out = f.create_dataset("p_"+sess, (data_file_size*NUM_REPEATS,NUM_POINTS,2), dtype='i')                           # point pos in 2D
    s_set_out = f.create_dataset("s_"+sess, (data_file_size*NUM_REPEATS,NUM_POINTS,1), dtype='i')                             # point labels, digits
    x_set_out = f.create_dataset("x_"+sess, (data_file_size*NUM_REPEATS,NUM_POINTS,3), dtype='f')                           # point pos in 3D


    idx_sample = 0
    node_loss_rate_list = []
    time_begin = time.time_ns()

    ##%% load original dataset
    x_set = f0['x_'+sess][:]
    s_set = f0['s_'+sess][:]
    y_set = f0['y_'+sess][:]

    for i_repeat in range(NUM_REPEATS):

        for idx in range(data_file_size):

            current_sample_data  = x_set[idx]
            current_sample_seg   = s_set[idx]
            current_sample_label = y_set[idx]

            ##%% rotation and jittering code from PointNet
            final_data = current_sample_data[np.newaxis,:,:]
            if(FLAG_ROTATION and sess == 'train' ):
                final_data = provider.rotate_point_cloud(final_data)
            if(FLAG_JITTER and sess == 'train' ):
                final_data = provider.jitter_point_cloud(final_data)

            ##%% 3D to 2D projection
            pos, node_loss_rate = GPGL2_seg(final_data[0],current_sample_seg)

            y_set_out[idx_sample] = current_sample_label
            p_set_out[idx_sample] = pos
            s_set_out[idx_sample] = current_sample_seg[:,np.newaxis]
            x_set_out[idx_sample] = current_sample_data

            print(file_name+":"+sess+": idx="+str(idx_sample)+"/"+str(len(x_set_out)),"node_loss_rate="+str(node_loss_rate))
            idx_sample+=1
            node_loss_rate_list.append(node_loss_rate)

    time_end = time.time_ns()
    node_loss_rate_final =np.array(node_loss_rate_list).mean()
    f.attrs['NUM_REPEATS']=NUM_REPEATS
    f.attrs['node loss ratio']=node_loss_rate_final
    f0.close()
    return idx_sample,node_loss_rate_final,time_end-time_begin


#%% main call function

#%% define dataset name
file_name ='ShapeNet_prepro.hdf5'
##%% create training and testing sets
f = h5py.File(file_name, 'w')
train_sample,train_node_loss,train_time = parepare_dataset('train',NUM_REPEATS,file_name)
val_sample,val_node_loss,val_time = parepare_dataset('val',1,file_name)
test_sample,test_node_loss,test_time = parepare_dataset('test',1,file_name)
f.close()


#%% output logs
print("train_sample:",train_sample,"train_node_loss:",train_node_loss,"train_time:",train_time/train_sample/1e6,"ms/sample")
print("val_sample:",val_sample,"val_node_loss:",val_node_loss,"val_time:",val_time/val_sample/1e6,"ms/sample")
print("test_sample:",test_sample,"test_node_loss:",test_node_loss,"test_time:",test_time/test_sample/1e6,"ms/sample")
wall_clock_end = time.time()
print('Dataset perpration time:',wall_clock_end - wall_clock_start,'s.')


