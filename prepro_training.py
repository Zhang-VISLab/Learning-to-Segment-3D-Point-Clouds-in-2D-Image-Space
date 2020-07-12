import h5py
import numpy as np


import tensorflow as tf
print(tf.version.VERSION)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess= tf.compat.v1.Session(config=config)

import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

# load GPGL functions
#%% global settings
NUM_POINTS = 2048
NUM_CUTS = 32
SIZE_SUB = 16
SIZE_TOP = 16

file_name = 'prepro_2048pts_32cuts_11x_r_j'
dataset_path = file_name+'.hdf5'
save_path = 'ShapeNet_model.h5'

#%%
f = h5py.File(dataset_path,'r')

print("loading training data")
x_train = f['x_train']
s_train = f['s_train']

print("loading testing data")
x_test  = f['x_test']
s_test  = f['s_test']

#%%
x_train_shape = x_train.shape
x_test_shape =  x_test.shape
print('train samples: %d, test samples: %d' % (x_train_shape[0], x_test_shape[0]))

#%%
class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_set, s_set, shuffle=0):
        self.x_set = x_set
        self.s_set = s_set
        self.size = len(s_set)
        self.indexes = np.arange(self.size)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        idx_temp = self.indexes[index]

        x_data = self.x_set[idx_temp:idx_temp+1]
        s_data = self.s_set[idx_temp:idx_temp+1]

        return x_data, s_data

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

#%%
train_generator = DataGenerator(x_train,s_train,1)
test_generator = DataGenerator(x_test,s_test,0)

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
opti = keras.optimizers.Adam(1e-4)
model.compile(opti,loss ='categorical_crossentropy',metrics=[weighted_acc])

print(model.summary())


#%%
print("-------------------------------")
print("*******************************")
print("*****",save_path,"*****")
print("*******************************")
print("-------------------------------")
checkpointer = keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_weighted_acc', verbose=1, save_best_only=True,save_weights_only=False)
namecall =  keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch,logs: print("*****",save_path,"*****"))


history = model.fit(x=train_generator, validation_data=test_generator,
        epochs = 30, verbose=1, callbacks=[checkpointer,namecall])

print("Best pixel wise accuracy",max(history.history['val_weighted_acc']))

#%%
