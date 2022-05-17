# Animal-Classification
This project aims to use computer vision to identify trap images with animals in them. These trap images were captured from various forests in Singapore and it was extremely time consuming to go through all the images manually to look for the ones we need for our research. To save time, we created this project to label our images for us. This computer vision modal was trained on our own site images. 

## Getting Started
### Prerequisites
+ Python 3.7
+ Cuda

### Installation
1. Install python packages
```
pip install -r requirements.txt
```

## Usage
Move your images to data/images folder. After that, update labels.csv inside the data directory with the correct labels for each image. labels.csv should be in the following format
```
FileName,SpeciesCode
<filename>.jpg,<label>
<filename>.jpg,<label>
<filename>.jpg,<label>
```

For the following steps, run from the src directory.

Verify that your images are all valid using the following command
```
python3 corrupt_imgs.py
```

Once you verify that all your images can be opened, resize the images with the following command so that training can be sped up
```
python3 resize.py
```

After that is done, create your own train, validation and test splits
```
python3 dataset.py
```

To train and test your model, run
```
python3 train.py
```

Finally, to label new data using the model you have created, run
```
python3 test.py 
```

Filepath configurations are mostly written inside config.py. Do open it and make adjustments if needed. train.py and test.py offers several command line arguments for you to adjust hyperparameters and set filepaths as well. Look at the source code to figure out what options are available. 
