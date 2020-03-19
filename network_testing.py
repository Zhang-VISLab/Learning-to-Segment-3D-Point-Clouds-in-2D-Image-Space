import os
# os.environ["CUDA_VISIBLE_DEVICES"]="";
import h5py
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

config = tf.ConfigProto()
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
save_path = file_name+'_NUM_POINTS_'+str(NUM_POINTS)+'NUM_CUTS_'+str(NUM_CUTS)+'SIZE_SUB_'+str(SIZE_SUB)+'SIZE_TOP_'+str(SIZE_TOP)

model_path = 'ShapeNet_training_NUM_POINTS_2048NUM_CUTS_32SIZE_SUB_16SIZE_TOP_16_model.h5'
f = h5py.File(dataset_path, 'r')

#%%
print("loading testing data")
x_test  = tf.keras.utils.HDF5Matrix(dataset_path,'x_test')
y_test  = np.array(tf.keras.utils.HDF5Matrix(dataset_path,'y_test')).argmax(-1)
s_test  = tf.keras.utils.HDF5Matrix(dataset_path,'s_test')
p_test = np.array(f['p_test'])
l_test  = tf.keras.utils.HDF5Matrix(dataset_path,'l_test')
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
    # x2 = L.Conv2D(filters,(3,3),padding='same',activation='linear')(input_tensor)
    # x3 = L.Conv2D(filters,(3,3),padding='same',activation='linear')(x2)
    x = L.Concatenate()([x0,x1])
    # x = L.Maximum()([x0,x1])
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
opti = keras.optimizers.Adam(1e-5)
model.compile(opti,loss ='categorical_crossentropy',metrics=[weighted_acc])


model.load_weights(model_path)

print(model.summary())
#%%
mIoU_class_list=[]
mIoU_instance_list = []

# for each class, get the CAD model list, and corresponding point-image lookup table
for i_class in range(16):
    idx_list = np.where(y_test==i_class)[0]
    pos_list = p_test[idx_list]
    gt_list  = l_test[idx_list]

    label_min = gt_list.min()
    label_max = gt_list.max()+1


    mIoU_class = 0
    
	# for each CAD model, get the point-image lookup table, ground truth in each class
    for pos,gt,idx in zip(pos_list,gt_list,idx_list):
		
		# predict one CAD model
        prediction = model.predict(x_test[idx:idx+1],batch_size = 1)[0]
        tp = 0
        fp = 0
        fn = 0
		
		# count the true positive, false positive and false negative samples in each CAD model
		#
		# P.S. 	Here tp,fp and fn are counted in CAD model instance level with multiple segmentation labels.
		# 		In instance-level counting, a fp in one class definitely leads to a false negative in one of the 
		#		other class. 
		#		For example, a CAD model of "Car" has three segment classes "body", "wheel" and "mirror", and at
		#		point A within class "body" is predicted as "wheel". At this time, a false positive is counted
		#		in "wheel"; meanwhile, a false negative is also counted in "body". 
        for p,g in zip(pos,gt):
            pre = prediction[p[0],p[1],label_min:label_max].argmax()+label_min
            tp += g==pre
            fp += g!=pre
            fn += g!=pre



		# count IoU of each CAD model
        mIoU_instance = tp/(tp+fp+fn)
        mIoU_instance_list.append(mIoU_instance)
        mIoU_class+=mIoU_instance

    mIoU_class_mean =mIoU_class/len(idx_list)
    mIoU_class_list.append(mIoU_class_mean)
    print('class',i_class,",acc=",mIoU_class_mean)


print("-----------")
print('mIoU_class_mean=',np.array(mIoU_class_list).mean())
print('mIoU_instance_list=',np.array(mIoU_instance_list).mean())
