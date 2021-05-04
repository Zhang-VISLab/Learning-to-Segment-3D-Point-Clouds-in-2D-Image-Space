import h5py
import numpy as np


import tensorflow as tf
print(tf.version.VERSION)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess= tf.compat.v1.Session(config=config)

import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

#%% load GPGL functions
import fun_network as fun
#%% global settings
NUM_POINTS = 2048
NUM_CUTS = 32
SIZE_SUB = 16
SIZE_TOP = 16
SIZE_IMG = SIZE_SUB*SIZE_SUB

dataset_path = 'ShapeNet_prepro.hdf5'
save_path = 'ShapeNet_model.h5'

epochs = 100
#%%
f = h5py.File(dataset_path,'r')

#%% initialize the data loader
train_loader = fun.Dataloder(f,'train', SIZE_IMG, batch_size=4,shuffle=1)
val_loader = fun.Dataloder(f,'val', SIZE_IMG, batch_size=1,shuffle=0)

#%%
input_shape = [SIZE_IMG,SIZE_IMG,3]
inputs1 = L.Input(shape=input_shape)
x = inputs1

mask = L.Lambda(lambda x: fun.build_mask(x),name='mask')(inputs1)
mask = L.Lambda(lambda x: K.expand_dims(x),name='expend')(mask)
mask1 = L.Lambda(lambda x: K.tile(x, (1, 1, 1, 50)),name='tile')(mask)
x0 = fun.Inception(x,64,name='x0')(x)

x1 = L.MaxPool2D(pool_size=(SIZE_SUB,SIZE_SUB),strides=(SIZE_SUB,SIZE_SUB),padding='valid')(x0)

x1 = fun.Inception(x1,128,name='x1')(x1)
x2 = L.MaxPool2D(pool_size=(SIZE_TOP,SIZE_TOP),strides=(SIZE_TOP,SIZE_TOP),padding='valid')(x1)
xg = x2


xg = L.Dense(256,activation='relu',name='x2')(xg)

y2 = xg
y1 = L.UpSampling2D(size=(SIZE_TOP, SIZE_TOP))(y2)
y1 = L.Concatenate()([x1,y1])
y1 = fun.Inception(y1,128,name='y1')(y1)

y0 = L.UpSampling2D(size=(SIZE_SUB, SIZE_SUB))(y1)
y0 = L.Concatenate()([x0,y0])
y0 = fun.Inception(y0,64,name='y0')(y0)

y = L.Dense(50,activation='softmax')(y0)


ouputs = y
ouputs = L.Multiply()([ouputs,mask1])
not_mask = L.Lambda(lambda x: 1-x)(mask)

ouputs = L.Concatenate(name="segment_out")([ouputs,not_mask])
model = keras.Model(inputs=inputs1, outputs=[ouputs])
print(model.summary())
#%%
opti = keras.optimizers.Adam(1e-4)
model.compile(opti,loss ='categorical_crossentropy',metrics=fun.iou)

#%%
print("-------------------------------")
print("*******************************")
print("*****",save_path,"*****")
print("*******************************")
print("-------------------------------")
checkpointer = keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_iou',mode='max', verbose=1, save_best_only=True,save_weights_only=False)


history = model.fit(x=train_loader, validation_data=val_loader,
        epochs = epochs, verbose=1, callbacks=[checkpointer])

print("Best pixel wise accuracy",max(history.history['iou']))

np.savetxt("ShapeNet_training_statistics.csv", np.vstack((history.history['loss'],history.history['iou'],history.history['val_loss'],history.history['val_iou'])).T , delimiter=",")

