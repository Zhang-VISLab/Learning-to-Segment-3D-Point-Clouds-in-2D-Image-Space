import os
# os.environ["CUDA_VISIBLE_DEVICES"]="";
import h5py
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess= tf.compat.v1.Session(config=config)

#%% global settings
NUM_POINTS = 2048
NUM_CUTS = 32
SIZE_SUB = 16
SIZE_TOP = 16
batch_size = 1

#%%


file_name = 'ShapeNet_testing'
dataset_path = file_name+'.hdf5'
result_path = file_name+'_result.hdf5'
model_path = 'ShapeNet_model.h5'
f = h5py.File(dataset_path, 'r')

#%%
print("loading testing data")
f = h5py.File(dataset_path,'r')
x_test = f['x_test']
y_test = f['y_test'][:]
p_test = f['p_test'][:]
l_test = f['l_test'][:]
d_test = f['d_test'][:]


#%%
class_weights = np.ones(51)
class_weights[-1] = 0

#%%
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


def MXM_2D(inputs,filters):
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

x0 = MXM_2D(x,64)(x)
x1 = L.MaxPool2D(pool_size=(SIZE_SUB,SIZE_SUB),strides=(SIZE_SUB,SIZE_SUB),padding='valid')(x0)

x1 = MXM_2D(x1,128)(x1)
x2 = L.MaxPool2D(pool_size=(SIZE_TOP,SIZE_TOP),strides=(SIZE_TOP,SIZE_TOP),padding='valid')(x1)
xg = x2


xg = L.Dense(256,activation='relu')(xg)

y2 = xg
y1 = L.UpSampling2D(size=(SIZE_TOP, SIZE_TOP))(y2)
y1 = L.Concatenate()([x1,y1])
y1 = MXM_2D(y1,128)(y1)

y0 = L.UpSampling2D(size=(SIZE_SUB, SIZE_SUB))(y1)
y0 = L.Concatenate()([x0,y0])
y0 = MXM_2D(y0,64)(y0)


y = L.Dense(50,activation='softmax')(y0)


ouputs = y
ouputs = L.Multiply()([ouputs,mask1])
not_mask = L.Lambda(lambda x: 1-x)(mask)

ouputs = L.Concatenate(name="segment_out")([ouputs,not_mask])
model = keras.Model(inputs=inputs1, outputs=[ouputs])

model.load_weights(model_path)

print(model.summary())

#%% get start and end label id for each object category
y_test_digits = np.argmax(y_test,-1)
class_label_region = np.zeros((16,2),dtype=np.int)
for i_class in range(16):
    idx_list = np.where(y_test_digits==i_class)[0]
    pos_list = p_test[idx_list]
    gt_list  = l_test[idx_list]

    label_min = gt_list.min()
    label_max = gt_list.max()

    class_label_region[i_class,0] = label_min
    class_label_region[i_class,1] = label_max


#%% create dataset to store the test result
test_set_len = len(x_test)
f1 = h5py.File(result_path,'w')
y_set = f1.create_dataset('y_test_digits',data = y_test_digits)
l_set = f1.create_dataset('l_test',data=l_test)
p_set = f1.create_dataset('p_test',data=p_test) # 2D position
d_set = f1.create_dataset('d_test',data=p_test) # 3D position
x_set = f1.create_dataset('pre_test', shape=(test_set_len,2048),dtype=np.int)


#%% transform image segments to point segments and store it to dataset
pre_test = np.zeros_like(l_test)
for idx_sample,pos,obj_class in zip(range(len(p_test)),p_test,y_test_digits):
    pre_image = model.predict(x_test[idx_sample:idx_sample+1],verbose=0,batch_size=1)[0]
    pre_sample = np.zeros_like(l_test[0])
    for id_point, pt in zip(range(len(pos)),pos):
        pre = pre_image[pt[0],pt[1]]

        label_min = class_label_region[obj_class,0]
        label_max = class_label_region[obj_class,1]

        pre = pre[label_min:label_max+1].argmax()+label_min
        pre_sample[id_point] = pre

    pre_test[idx_sample] = pre_sample
    x_set[idx_sample] = pre_sample
    if( idx_sample % 100 == 0):
        print('finish point segments: ',idx_sample,'/',len(l_test))

#%% close the result dataset
f.close()
f1.close()

#%% calculate iou for each shape
iou_shape = np.zeros(len(l_test))
for idx_sample,pre_sample,gt_sample,obj_class in zip(range(len(l_test)),pre_test,l_test,y_test_digits):
    label_min = class_label_region[obj_class,0]
    label_max = class_label_region[obj_class,1]

    iou_list = []
    # % for each segment, calculate iou
    for i_class in range(label_min,label_max+1):
        # break
        tp = np.sum( (pre_sample == i_class) * (gt_sample == i_class) )
        fp = np.sum( (pre_sample == i_class) * (gt_sample != i_class) )
        fn = np.sum( (pre_sample != i_class) * (gt_sample == i_class) )

        # % if current sugment exists then count the iou
        if(tp+fp+fn>0):
            iou = tp / (tp+fp+fn)
        else:
            iou=1

        iou_list.append(iou)

    iou_shape[idx_sample] = np.mean(iou_list)

    if( idx_sample % 100 == 0):
        print('finish iou cauculation: ',idx_sample,'/',len(l_test))

print( 'iou_instacne =', iou_shape.mean())
#%% calculate iou for each class
iou_class = np.zeros(16)
for obj_class in range(16):
    iou_obj_class = iou_shape[y_test_digits==obj_class]
    iou_class[obj_class] = iou_obj_class.mean()
print( 'iou_class =', iou_class.mean())

for obj_class in range(16):
    print('class',obj_class,",iou=",iou_class[obj_class])