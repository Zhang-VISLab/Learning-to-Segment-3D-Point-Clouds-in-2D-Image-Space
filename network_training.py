# import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1";  
import h5py
import numpy as np

import keras
import keras.layers as L
import keras.backend as K
from keras.utils import HDF5Matrix
from sklearn.cluster import KMeans
from scipy.spatial.distance  import cdist,pdist,squareform
import tensorflow as tf
import networkx as nx
print(tf.version.VERSION)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess= tf.compat.v1.Session(config=config)

# load GPGL functions
import provider
from fun_GPGL import fun_GPGL_layout_push,graph_cut,fun_graph_cosntruct
#%% global settings
NUM_POINTS = 2048
NUM_CUTS = 32
SIZE_SUB = 16
SIZE_TOP = 16
batch_size = 1


file_name = 'ShapeNet_training'
dataset_path = file_name+'.hdf5'
save_path = file_name+'_NUM_POINTS_'+str(NUM_POINTS)+'NUM_CUTS_'+str(NUM_CUTS)+'SIZE_SUB_'+str(SIZE_SUB)+'SIZE_TOP_'+str(SIZE_TOP)

#%% Kmeans solver initialization

kmeans_solver = KMeans(n_clusters=NUM_CUTS, n_init=1,n_jobs=1,max_iter=100)

#%%
print("loading training data")
x_train = np.array(tf.keras.utils.HDF5Matrix(dataset_path,'x_train')[:])
s_train = np.array(tf.keras.utils.HDF5Matrix(dataset_path,'s_train')[:])


#%%
print("loading testing data")
x_test  = np.array(tf.keras.utils.HDF5Matrix(dataset_path,'x_test')[:2874])
s_test  = np.array(tf.keras.utils.HDF5Matrix(dataset_path,'s_test')[:2874])
#%%
class_weights = np.ones(51)
class_weights[-1] = 0
#%%

x_train_shape = x_train.shape
x_test_shape =  x_test.shape
print('train samples: %d, test samples: %d' % (x_train_shape[0], x_test_shape[0]))


#%%
def GPGL2_seg(data,current_sample_seg):
    data = data+np.random.rand(len(data),len(data[0]))*1e-6
    dist_mat = kmeans_solver.fit_transform(data)


    node_top,labels = graph_cut(data,dist_mat,NUM_POINTS,NUM_CUTS)

    aij_mat = fun_graph_cosntruct(node_top)

    # aij_mat = squareform(pdist(node_top),checks=False)
    H = nx.from_numpy_matrix(aij_mat)
    pos_spring = nx.spring_layout(H)
    pos_spring = np.array([pos for idx,pos in sorted(pos_spring.items())])
    # pos_spring = np.array(forceatlas2.forceatlas2(aij_mat, None, 10))

    pos = fun_GPGL_layout_push(pos_spring,SIZE_SUB)
    pos_top = fun_GPGL_layout_push(pos_spring,SIZE_TOP)

    ##%%
    pos_cuts = []
    for i_cut in range(NUM_CUTS):
        pos_cut_3D = data[labels==i_cut,:]

        if(len(pos_cut_3D)<5):
            pos_raw = [[0,0],[0,1],[1,1],[1,0]]
            pos = pos_raw[:len(pos_cut_3D)]
            # H =  nx.from_numpy_matrix(np.ones([len(pos_cut_3D),len(pos_cut_3D)]))
            pos_cuts.append(pos)
            continue
        aij_mat = fun_graph_cosntruct(pos_cut_3D)
        # aij_mat = squareform(pdist(pos_cut_3D),checks=False)
        H = nx.from_numpy_matrix(aij_mat)
        pos_spring = nx.spring_layout(H)
        pos_spring = np.array([pos for idx,pos in sorted(pos_spring.items())])
        # pos_spring = np.array(forceatlas2.forceatlas2(aij_mat, None, 10))

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

    return mat, seg



#%%
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_set, s_set, batch_size=1,FLAG_ROTATION=0,FLAG_JITTER=0,shuffle=0):
        self.x_set = x_set
        self.s_set = s_set
        self.size = len(s_set)
        self.indexes = np.arange(self.size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.FLAG_JITTER = FLAG_JITTER
        self.FLAG_ROTATION =FLAG_ROTATION
        self.on_epoch_end()

    def __len__(self):
        #return int(np.floor(self.size / self.batch_size))
        return self.size//self.batch_size

    def __getitem__(self, index):
        idx_temp = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        current_sample_data = self.x_set[idx_temp]
        label_batch = self.s_set[idx_temp]

        final_data = current_sample_data[:,:NUM_POINTS,:]


        if(self.FLAG_ROTATION):
            final_data = provider.rotate_point_cloud(final_data)
        if(self.FLAG_JITTER):
            final_data = provider.jitter_point_cloud(final_data)

        data_batch = np.zeros([len(label_batch),SIZE_SUB*SIZE_TOP,SIZE_SUB*SIZE_TOP,3])
        seg_batch  = np.zeros([len(label_batch),SIZE_SUB*SIZE_TOP,SIZE_SUB*SIZE_TOP,51])
        for idx in range(len(label_batch)):
            data_batch[idx],seg_batch[idx]=GPGL2_seg(final_data[idx],label_batch[idx])


        return data_batch, seg_batch

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

#%%

train_generator = DataGenerator(x_train,s_train,batch_size,FLAG_ROTATION=1,FLAG_JITTER=1,shuffle=1)
test_generator = DataGenerator(x_test,s_test,1)

#%%
def weighted_acc(y_true, y_pred):
    y_true_digit = K.argmax(y_true,axis=-1)
    y_pred_digit = K.argmax(y_pred,axis=-1)
    mask = tf.subtract(K.constant(1.0,dtype=tf.float32),y_pred[:,:,:,-1])
    true_mat = K.cast(K.equal(y_true_digit,y_pred_digit),K.floatx())

    tp = K.sum(K.minimum(true_mat,mask))
    fp = K.sum(K.minimum(1-true_mat,mask))
    fn = K.sum(K.minimum(1-true_mat,mask))

    return tp/(tp+fp+fn+1e-12)

def build_mask(input):
    mask_mat = K.stop_gradient(K.abs(input))
    mask = K.stop_gradient(K.sum(mask_mat,axis=-1,keepdims=False))
    mask = K.stop_gradient(K.sign(mask))
    return mask


def Inception(inputs,filters):
    input_shape = inputs.shape[1:].as_list()
    input_tensor = L.Input(shape=input_shape)
    x0 = L.Conv2D(filters,(1,1),padding='same',activation='linear')(input_tensor)
    x1 = L.Conv2D(filters,(3,3),padding='same',activation='linear')(input_tensor)
    x = L.Concatenate()([x0,x1])
    x = L.ReLU()(x)
    model = keras.Model(inputs=input_tensor, outputs=x)
    return model



#%%
input_shape = [SIZE_SUB*SIZE_TOP,SIZE_SUB*SIZE_TOP,3]
inputs1 = L.Input(shape=input_shape)
x = inputs1

mask = L.Lambda(build_mask,[SIZE_SUB*SIZE_TOP,SIZE_SUB*SIZE_TOP])(inputs1)
mask = L.Lambda(K.expand_dims,name = 'expend')(mask)
mask1 = L.Lambda(K.tile, arguments={'n':(1, 1, 1, 50)},name='tile')(mask)

x0 = Inception(x,64)(x)
x1 = L.MaxPool2D(pool_size=(SIZE_SUB,SIZE_SUB),strides=(SIZE_SUB,SIZE_SUB),padding='valid')(x0)

x1 = Inception(x1,128)(x1)
x2 = L.MaxPool2D(pool_size=(SIZE_TOP,SIZE_TOP),strides=(SIZE_TOP,SIZE_TOP),padding='valid')(x1)
xg = x2


xg = L.Dense(256,activation='relu')(xg)

y2 = xg
y1 = L.UpSampling2D(size=(SIZE_TOP, SIZE_TOP))(y2)
y1 = L.Concatenate()([x1,y1])
y1 = Inception(y1,128)(y1)

y0 = L.UpSampling2D(size=(SIZE_SUB, SIZE_SUB))(y1)
y0 = L.Concatenate()([x0,y0])
y0 = Inception(y0,64)(y0)


y = L.Dense(50,activation='softmax')(y0)


ouputs = y
ouputs = L.Multiply()([ouputs,mask1])
not_mask = L.Lambda(lambda x: 1-x)(mask)

ouputs = L.Concatenate(name="segment_out")([ouputs,not_mask])
model = keras.Model(inputs=inputs1, outputs=[ouputs])
opti = keras.optimizers.Adam(1e-5)
model.compile(opti,loss ='categorical_crossentropy',metrics=[weighted_acc])

print(model.summary())
#%%
print("-------------------------------")
print("*******************************")
print("*****",save_path,"*****")
print("*******************************")
print("-------------------------------")
# file_path = save_path+'_model.h5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath=save_path+"_model.h5", monitor='val_weighted_acc', verbose=1, save_best_only=True,save_weights_only=False)
namecall =  keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch,logs: print("*****",save_path,"*****"))


history = model.fit_generator(generator = train_generator,class_weight = [class_weights],
                              steps_per_epoch=x_train_shape[0]//batch_size,
                              use_multiprocessing=False,
                              epochs = 100,  verbose=1,
                              validation_data=test_generator,
                              validation_steps=x_test_shape[0],
                               callbacks=[checkpointer,namecall]
                              )

print("Best pixel wise accuracy",max(history.history['val_weighted_acc']))

