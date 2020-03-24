import h5py
import numpy as np
import time
import threading

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


from tensorflow import keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

#%% Training and testing dataset name
file_name = 'PartNet_0_Knife-1_1x_j'
training_epochs = 30
#%%
dataset_path = file_name+'.hdf5'
save_path = file_name+'_Inception'
file_path = save_path+'_model.h5'
f = h5py.File(dataset_path, 'r')
#%% load training set
print("loading training data")
x_train = f['x_train']
s_train = f['s_train']
#%% load testing set
print("loading testing data")
x_test = f['x_test']
l_test = f['l_test']
s_test = f['s_test']
p_test = f['p_test']
#%%
NUM_CLASS =s_train.shape[-1]-1
class_weights = np.ones(NUM_CLASS+1)
class_weights[-1] = 0
#%%
x_train_shape = x_train.shape
x_test_shape =  x_test.shape
print('train samples: %d, test samples: %d' % (x_train_shape[0], x_test_shape[0]))

#%% help functions
def weighted_acc(y_true, y_pred):
    y_true_digit = K.argmax(y_true,axis=-1)
    y_pred_digit = K.argmax(y_pred,axis=-1)
    mask = tf.subtract(K.constant(1.0,dtype=tf.float32),y_pred[:,:,:,-1])
    true_mat = K.cast(K.equal(y_true_digit,y_pred_digit),K.floatx())
    return K.sum(tf.multiply(true_mat,mask))/K.sum(mask+1e-12)

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



#%% define the network
input_shape = [x_train_shape[1],x_train_shape[2],x_train_shape[3]]
inputs1 = L.Input(shape=input_shape)
x = inputs1

mask = L.Lambda(build_mask,[x_train_shape[1],x_train_shape[2]])(inputs1)
mask = L.Lambda(K.expand_dims,name = 'expend')(mask)
mask1 = L.Lambda(tf.tile, arguments={'multiples':(1, 1, 1, NUM_CLASS)},name='tile')(mask)

x0 = Inception(x,64)(x)
x1 = L.MaxPool2D(pool_size=(16,16),strides=(16,16),padding='valid')(x0)

x1 = Inception(x1,128)(x1)
x2 = L.MaxPool2D(pool_size=(16,16),strides=(16,16),padding='valid')(x1)
xg = x2


xg = L.Dense(256,activation='relu')(xg)

y2 = xg
y1 = L.UpSampling2D(size=(16, 16))(y2)
y1 = L.Concatenate()([x1,y1])
y1 = Inception(y1,128)(y1)

y0 = L.UpSampling2D(size=(16, 16))(y1)
y0 = L.Concatenate()([x0,y0])
y0 = Inception(y0,64)(y0)


y = L.Dense(NUM_CLASS,activation='softmax')(y0)


ouputs = y
ouputs = L.Multiply()([ouputs,mask1])
not_mask = L.Lambda(lambda x: 1-x)(mask)

ouputs = L.Concatenate(name="segment_out")([ouputs,not_mask])
model = keras.Model(inputs=inputs1, outputs=[ouputs])
opti = keras.optimizers.Adam(1e-4)
model.compile(opti,loss ='categorical_crossentropy',metrics=[weighted_acc])



# model.load_weights(file_path)
print(model.summary())
#%% train the network
print("-------------------------------")
print("*******************************")
print("*****",save_path,"*****")
print("*******************************")
print("-------------------------------")

checkpointer = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_weighted_acc', verbose=1, save_best_only=True,save_weights_only=False)
namecall =  keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch,logs: print("*****",save_path,"*****"))


history = model.fit(x = x_train, y= [s_train], batch_size = 1,shuffle="batch", epochs = training_epochs,  verbose=1,
                    validation_data=(x_test, [s_test]),callbacks=[checkpointer,namecall],class_weight = [class_weights])


#%% use trained network to predict the image segments
prediction_all = model.predict(x_test,verbose=1,batch_size=1)

#%% transform image segments to point segments
pre_test = np.zeros_like(l_test)
for idx_sample,pos,pre_image in zip(range(len(p_test)),p_test,prediction_all):

    pre_sample = np.zeros_like(l_test[0])
    for id_point, pt in zip(range(len(pos)),pos):
        pre = pre_image[pt[0],pt[1]]

        pre = pre.argmax()
        pre_sample[id_point] = pre

    pre_test[idx_sample] = pre_sample

#%% calculate iou for each shape
miss_count = 0
iou_shape = np.zeros(len(l_test))
for idx_sample,pre_sample,gt_sample in zip(range(len(l_test)),pre_test,l_test):


    iou_list = []
    # % for each segment, calculate iou
    for i_class in range(np.max(l_test)):
        tp = np.sum( (pre_sample == i_class) * (gt_sample == i_class) )
        fp = np.sum( (pre_sample == i_class) * (gt_sample != i_class) )
        fn = np.sum( (pre_sample != i_class) * (gt_sample == i_class) )

        # % if current sugment exists then count the iou
        if(tp+fp+fn>0):
            iou = tp / (tp+fp+fn)
            iou_list.append(iou)



    iou_shape[idx_sample] = np.mean(iou_list)

    if( idx_sample % 100 == 99):
        print('finish iou cauculation: ',idx_sample,'/',len(l_test))

print( 'iou_shape =', np.nanmean(iou_shape)*100,'%')