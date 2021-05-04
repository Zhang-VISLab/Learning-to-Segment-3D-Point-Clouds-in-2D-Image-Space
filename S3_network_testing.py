import h5py
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess= tf.compat.v1.Session(config=config)
#%%
import fun_network as fun
#%% global settings
NUM_POINTS = 2048
NUM_CUTS = 32
SIZE_SUB = 16
SIZE_TOP = 16
SIZE_IMG = SIZE_SUB*SIZE_SUB
batch_size = 1

#%%
dataset_path = 'ShapeNet_prepro.hdf5'
result_path = 'ShapeNet_testing_result.hdf5'
model_path = 'ShapeNet_model.h5'
f = h5py.File(dataset_path, 'r')
class_name = np.genfromtxt('hdf5_data/all_object_categories.txt',dtype='U')[:,0]
#%%
print("loading testing data")
f = h5py.File(dataset_path,'r')
x_test = f['x_test'][:]
y_test = f['y_test'][:]
s_test = f['s_test'][:]
p_test = f['p_test'][:]


#%%
test_loader  = fun.Dataloder(f,'test', SIZE_IMG, batch_size=1,shuffle=0)

#%%
model = tf.keras.models.load_model(model_path,custom_objects={'iou':fun.iou})
print(model.summary())

#%% get start and end label id for each object category
class_label_region = np.zeros((16,2),dtype=np.int)
for i_class in range(16):
    idx_list = np.where(y_test==i_class)[0]
    gt_list  = s_test[idx_list]

    label_min = gt_list.min()
    label_max = gt_list.max()

    class_label_region[i_class,0] = label_min
    class_label_region[i_class,1] = label_max


#%% create dataset to store the test result
test_set_len = len(y_test)
f1 = h5py.File(result_path,'w')
x_set = f1.create_dataset('x_test',data = x_test) # point cloud position in 3D
y_set = f1.create_dataset('y_test',data = y_test) # point cloud shape class
s_set = f1.create_dataset('s_test',data = s_test) # point cloud segments
p_set = f1.create_dataset('p_test',data = p_test) # 2D position
pre_set = f1.create_dataset('pre_test', shape=(test_set_len,2048,1),dtype=np.int)


#%% transform image segments to point segments and store it to dataset
pre_test = np.zeros_like(s_test)
for idx_sample,pos,obj_class in zip(range(len(p_test)),p_test,y_test):
    input_tensor, output_tensor = test_loader.__getitem__(idx_sample)

    pre_image = model.predict(input_tensor,verbose=0,batch_size=1)[0]
    pre_sample = np.zeros_like(pre_test[0])

    label_min = int(class_label_region[obj_class,0])
    label_max = int(class_label_region[obj_class,1]+1)

    pre_image = pre_image[:,:,label_min:label_max].argmax(-1)+label_min
    pre_sample = pre_image[pos[:,0],pos[:,1]]



    pre_test[idx_sample] = pre_sample[:,np.newaxis]
    pre_set[idx_sample]  = pre_sample[:,np.newaxis]
    if( idx_sample % 100 == 0):
        print('finish point segments: ',idx_sample,'/',len(s_test))

#%% close the result dataset
f.close()
f1.close()

#%% calculate iou for each shape
iou_shape = np.zeros(len(s_test))
for idx_sample,pre_sample,gt_sample,obj_class in zip(range(len(s_test)),pre_test,s_test,y_test):
    label_min = int(class_label_region[obj_class,0])
    label_max = int(class_label_region[obj_class,1]+1)

    iou_list = []
    # % for each segment, calculate iou
    for i_class in range(label_min,label_max):
        # break
        tp = np.sum( (pre_sample == i_class) * (gt_sample == i_class) )
        fp = np.sum( (pre_sample == i_class) * (gt_sample != i_class) )
        fn = np.sum( (pre_sample != i_class) * (gt_sample == i_class) )

        # % if current sugment exists then count the iou
        iou = (tp+1e-12) / (tp+fp+fn+1e-12)


        iou_list.append(iou)

    iou_shape[idx_sample] = np.mean(iou_list)

    if( idx_sample % 100 == 0):
        print('finish iou cauculation: ',idx_sample,'/',len(s_test))

print( 'iou_instacne =', iou_shape.mean())
#%% calculate iou for each class
iou_class = np.zeros(16)
for obj_class in range(16):
    iou_obj_class = iou_shape[y_test[:,0]==obj_class]
    iou_class[obj_class] = iou_obj_class.mean()
print( 'iou_class =', iou_class.mean())

for obj_class in range(16):
    print('class',obj_class,', class name:',class_name[obj_class],",iou=",iou_class[obj_class])
