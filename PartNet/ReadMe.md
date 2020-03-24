# PartNet semantic segmentation


## Download dataset

Please turn to the PartNet offical download page at [https://www.shapenet.org/download/parts](https://www.shapenet.org/download/parts) and download the "HDF5 files for the semantic segmentation task (Sec 5.1 of PartNet paper)". You need to become a registered user in order to download the data.

After downloading, please unzip the dataset to /PartNet/sem_seg_h5 folder.

## Prepare the dataset for network training and testing
File **data_preparation_training_testing.py** is used to prepare dataset of each PartNet category for network training. Since PartNet contains huge amount of samples, the dataset is designed to run category by category. In **Line 27**, you can select the category to be prepared by changing the **category_id**. You can also adjust the global settings from **Line 15-21**  for your own purposes. When code is finished, an hdf5 file with the category id and category name will shown in the /PartNet folder.

## Training and testing the network
File **PartNet_train_testing.py** is used to train and test the network. In the file, **Line 21** defines the dataset file to train and test. Please put the name of the prepared dataset there without ".hdf5". After running the code, Mean IoU of this category will print on the console.