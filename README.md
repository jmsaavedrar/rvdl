# First steps

## Installing Tensorflow
I recommend installing Python and tensorflow using Anaconda ()
# OCR with MLP 
Let's start with an OCR example, where a MLP model is trained for classifying images like those appearing below:

## Preparing the data
We will need to define the training dataset and the validation dataset. In our case, both datasets can be downloaded from XXX. 
The datasets will be defined through two files. The first, train-txt, will show all the images used for training process. 
The second, val.txt, will define the images used for validation (or testing). You will note that each image is acompannied with its correspodning label.
Our OCR example consists of 12 different classes.
## Feature Extraction
We will use a vanilla histogram of orientations as feature vector. To compute the feature vector for each image we will use the following line:
$ ---
## Defining an architecture
We create a simple architecture MLP..
## Training
## Testing
