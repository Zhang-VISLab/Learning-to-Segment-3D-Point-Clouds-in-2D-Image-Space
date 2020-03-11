# Learning to Segment 3D Point Clouds in 2D Image Space

## conda environment setup
```
conda create -n XYZNet
conda activate XYZNet
conda install python=3.6 tensorflow-gpu=1.14 scikit-learn scipy networkx keras
```

## Download the ShapeNet part dataset
```
sh download_data.sh
```

## Prepare training and testing dataset
```
python data_preparation_training.py
python data_preparation_testing.py
```

## Train the network
```
python network_training.py
```

## Test the network
```
python network_testing.py
```
