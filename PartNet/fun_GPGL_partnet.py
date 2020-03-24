# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.sparse import csc_matrix
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
import networkx as nx




#%%


def fun_GPGL_layout_push(pos,size):
    dist_mat = pdist(pos)
    scale1 = 1/dist_mat.min()
    scale2 = (size-2)/(pos.max()-pos.min())
    scale = np.min([scale1,scale2])
    pos = pos*scale

    pos_quat = np.round(pos).astype(np.int)
    pos_quat = pos_quat-np.min(pos_quat,axis=0)+[1,1]
    pos_unique, count = np.unique(pos_quat,axis=0,return_counts=True)

    mask = np.zeros([size,size]).astype(np.int)
    for pt in pos_quat:
        mask[pt[0],pt[1]]+=1

    for i_loop in range(50):
        if(mask.max()<=1):
            # print("early stop")
            break
        idxs = np.where(count>1)[0]
        for idx in idxs:
            pos_overlap = pos_unique[idx]
            dist = cdist(pos_quat,[pos_overlap])
            idy = np.argmin(dist)

            b_down = np.maximum(pos_overlap[0]-1,0)
            b_up   = np.minimum(pos_overlap[0]+2,size)
            b_left = np.maximum(pos_overlap[1]-1,0)
            b_right= np.minimum(pos_overlap[1]+2,size)

            mask_target = mask[b_down:b_up,b_left:b_right]
            if(mask_target.min()==0):
                pos_target = np.unravel_index(np.argmin(mask_target),mask_target.shape)
                pos_mask = pos_target+np.array([b_down,b_left])

            else:
                pos_empty = np.array(np.where(mask==0)).T
                dist = cdist(pos_empty,[pos_overlap])
                pos_target  = pos_empty[np.argmin(dist)]
                direction = (pos_target-pos_overlap)
                direction1 = np.round(direction/np.linalg.norm(direction))
                pos_mask = pos_overlap+direction1.astype(np.int)


            pos_quat[idy]=pos_mask
            mask[pos_overlap[0],pos_overlap[1]] -=1
            mask[pos_mask[0],pos_mask[1]] +=1

            pos_unique, count = np.unique(pos_quat,axis=0,return_counts=True)

    return pos_quat




#%%
def graph_cut(data,dist_mat,NUM_POINTS,NUM_CUTS):
    NUM_CUTPOINTS = int(NUM_POINTS/NUM_CUTS)
    CUTPOINTS_THRESHOLD = np.ceil(NUM_CUTPOINTS*1.2)
    clsuter = np.argmin(dist_mat,axis=-1)
    mask = np.zeros([NUM_POINTS,NUM_CUTS])
    for m, c in zip(mask,clsuter):
        m[c]=1
    loss_mask = mask.sum(0)

    flow_mat = np.zeros([NUM_CUTS,NUM_CUTS])


    ## %% separate point cloud into NUM_CUTS clusters
    for i_loop in range(500):
        loss_mask = mask.sum(0)
        order_list = np.argsort(loss_mask)
        if(loss_mask.max()<=CUTPOINTS_THRESHOLD+1):
            break
        for i_order,order in zip(range(len(order_list)),order_list):
            if(loss_mask[order]>CUTPOINTS_THRESHOLD):
                idxs = np.where(mask[:,order])[0]
                idys_ori = order_list[:i_order]
                idys = []
                for idy in idys_ori:
                    if(flow_mat[order,idy]>=0):
                        idys.append(idy)

                mat_new = dist_mat[idxs,:]
                mat_new = mat_new[:,idys]
                cost_list_row = mat_new.argmin(-1)
                cost_list_col= mat_new.min(-1)

                row = cost_list_col.argmin(-1)
                col = cost_list_row[row]

                target_idx = [idxs[row],idys[col]]
                mask[target_idx[0],order]=0
                mask[target_idx[0],target_idx[1]]=1
                flow_mat[order,target_idx[1]]=1
                flow_mat[target_idx[1],order]=-1
    center_pos = []
    for i_cut in range(NUM_CUTS):
        if mask[:,i_cut].sum()>0:
            center_pos.append(data[mask[:,i_cut].astype(np.bool),:].mean(0))
        else:
            center_pos.append([0,0])
    labels = mask.argmax(-1)
    return np.array(center_pos),labels

#%%
def fun_graph_cosntruct(node_top):
    NUM_CUTS = len(node_top)
    node_top += np.random.rand(node_top.shape[0],node_top.shape[1])*1e-6
    tri = Delaunay(node_top)
    edges = np.vstack([tri.simplices[:,[0,1]],tri.simplices[:,[0,2]],tri.simplices[:,[0,3]],
                            tri.simplices[:,[1,2]],tri.simplices[:,[1,3]],tri.simplices[:,[2,3]]])
    aij_mat = csc_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])), shape=(NUM_CUTS, NUM_CUTS)).toarray()
    aij_mat = (aij_mat+aij_mat.T)>0
    return aij_mat



def GPGL2_seg(data,current_sample_seg, NUM_CLASSES, NUM_CUTS,SIZE_TOP,SIZE_SUB):
    NUM_POINTS = len(data)
    kmeans_solver = KMeans(n_clusters=NUM_CUTS, n_jobs=10,max_iter=100)
    data = data+np.random.rand(len(data),len(data[0]))*1e-6
    dist_mat = kmeans_solver.fit_transform(data)


    node_top,labels = graph_cut(data,dist_mat,NUM_POINTS,NUM_CUTS)
    aij_mat = fun_graph_cosntruct(node_top)

    H = nx.from_numpy_matrix(aij_mat)
    pos_spring = nx.spring_layout(H)
    pos_spring = np.array([pos for idx,pos in sorted(pos_spring.items())])

    pos = fun_GPGL_layout_push(pos_spring,SIZE_SUB)
    pos_top = fun_GPGL_layout_push(pos_spring,SIZE_TOP)

    #%%
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


    # %% combine all layout positions
    cuts_count = np.zeros(NUM_CUTS).astype(np.int)
    pos_all = []
    for idx in range(NUM_POINTS):
        label = labels[idx]
        pos_all.append(pos_cuts[label][cuts_count[label]]+pos_top[label]*SIZE_SUB)
        cuts_count[label] +=1
    poss_all=np.array(pos_all)




    #%% assign all features into the grid map
    mat = np.zeros([SIZE_SUB*SIZE_TOP,SIZE_SUB*SIZE_TOP,3])
    seg = np.zeros([SIZE_SUB*SIZE_TOP,SIZE_SUB*SIZE_TOP,NUM_CLASSES+1])
    seg[:,:,-1] = 1

    for data1,seg1, pos in zip(data,current_sample_seg,pos_all):
        mat[pos[0],pos[1]]=data1
        seg[pos[0],pos[1],int(seg1)]=1
        seg[pos[0],pos[1],-1]=0


    num_nodes_m = np.sum(np.linalg.norm(mat,axis=-1)>0)
    node_loss_rate=(1- num_nodes_m/NUM_POINTS)
    return mat, seg, poss_all, node_loss_rate