# Animal-Classification
Playing around with pytorch and experimenting with transfer learning. I am using my own set of images for this experimentation but I am not uploading the images as there are too many of them.

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
Move your images to data/images folder. After that, update labels.csv with the correct labels for each image. labels.csv should be in the following format

```
<filename>.jpg,<label>
<filename>.jpg,<label>
<filename>.jpg,<label>
```

To create your own train, validation and test splits, run

```
python3 dataset.py
```

To train and test your model, run
```
python3 run.py
```

Most of the code related to splitting the data is stored in dataset.py while all the code related to the model is stored in run.py. Feel free to update the hyperparameters in run.py to tune the model. 

## Contributing
Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
3. Commit your Changes (git commit -m 'Add some AmazingFeature')
4. Push to the Branch (git push origin feature/AmazingFeature)
5. Open a Pull Request
