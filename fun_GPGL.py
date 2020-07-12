import numpy as np
from scipy.spatial.distance import cdist,pdist
from scipy.sparse import csc_matrix
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
    # node_loss = np.sum(count)-len(count)
    # print('node_loss',node_loss)

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




def pc_to_cmatrix(current_sample_data):
    G = nx.Graph()
    for idx,pt in zip(range(len(current_sample_data)),current_sample_data):
        G.add_node(idx,x=pt[0],y=pt[1],z=pt[2])

    if(current_sample_data[:,2].max()==current_sample_data[:,2].min()):
        pos_ball =current_sample_data[:,:2]
        tri = Delaunay(pos_ball)
        for edge_list  in tri.simplices:
            for idx0 in range(len(edge_list)):
                for idx1 in range(idx0+1,len(edge_list)):
                    node0 = edge_list[idx0]
                    node1 = edge_list[idx1]
                    G.add_edge(node0,node1)

    else:
        tri = Delaunay(current_sample_data + np.random.rand(current_sample_data.shape[0],current_sample_data.shape[1])*1e-4)
        for node_list  in tri.simplices:
            for idx0 in range(len(node_list)):
                node0 = node_list[idx0]
                node_pos0 = current_sample_data[node0]
                node_list_rest =np.hstack([node_list[:idx0],node_list[idx0+1:]])
                for idx1 in range(len(node_list_rest)-1):
                    node1 = node_list_rest[idx1]
                    node2 = node_list_rest[idx1+1]
                    node_pos1 = current_sample_data[node1]
                    node_pos2 = current_sample_data[node2]

                    l01 = np.dot(node_pos1-node_pos0,node_pos1-node_pos0)
                    l02 = np.dot(node_pos2-node_pos0,node_pos2-node_pos0)
                    l12 = np.dot(node_pos2-node_pos1,node_pos2-node_pos1)

                    if(l12<l01+l02 and l12<0.04):

                        G.add_edge(node0,node1)
    return nx.to_numpy_matrix(G)

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
    tri = Delaunay(node_top)
    edges1 = np.vstack([tri.simplices[:,[0,1]],tri.simplices[:,[0,2]],tri.simplices[:,[0,3]],
                            tri.simplices[:,[1,2]],tri.simplices[:,[1,3]],tri.simplices[:,[2,3]]])

    edges = edges1

    aij_mat = csc_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])), shape=(NUM_CUTS, NUM_CUTS)).toarray()
    aij_mat = (aij_mat+aij_mat.T)>0
    return aij_mat
