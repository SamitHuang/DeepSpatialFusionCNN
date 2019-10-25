## Update
change to be compatible with the resnet18 in PyTorch 1.2  
remove tensorboardX

## Config
pytorch 1.2.0 
dataset: ICIAR 2018, please download via the grand-challenge website. 

To split the dataset into 10-fold, please run the following commind in the "dataset2018" folder:
```
python split_dataset.py
```

## Single fold model
To train the patch-wise network:
```
$ python train.py --network=1 --patches-overlap=0 --fold-index=1
```
Config datset (2015/2018) and checkpoint path in option.py

To evaluate the patch-wise network:
```
$ python validate.py --network=1 --patches-overlap=0 --fold-index=1  
```

To train the image-wise network:
```
$ python train.py --network=2 --patches-overlap=0 --fold-index=1
```
Config datset and checkpoint path in option.py

To evaluate the image-wise netowork:
```
$ python validate.py --network=2 --patches-overlap=0 --fold-index=1
```

## Cross validation
To run the 10-fold cross-validatoin (based on the trained models):
```
$ python corss_evaluate.py
```

