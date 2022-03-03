# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. The dataset for this project is https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip. The goal of the project is to create a classifier capable of determining a dog's breed from a photo.


## Dataset
The dataset I'm using for this project is dog breed dataset (https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). After unzipping, there are three folders - train, valid and test, with each containing 133 subfolders, representing 133 dog breeds. The goal of the project is to create a classifier capable of determining a dog's breed from a photo.

## Hyperparameter Tuning
I used resnet18 as the pretrained model because it's one of the most widely used models for image classification. Below is a list of hyperparameters I chose to optimize for:

- Learning rate: ContinuousParameter(0.001, 0.1),
- Batch size: CategoricalParameter([32, 64, 128, 256, 512]),
- Epochs: CategoricalParameter([5,10,20,40,100,200])


Screenshot of completed training jobs:

![hyperparameter tuning](./screenshots/hpo_job.png)

![training job](./screenshots/training_job.png)


Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling

Steps to perform model debugging:

- In the train() function, add the SMDebug hook for PyTorch with TRAIN mode.
- In the test() function, add the SMDebug hook for PyTorch with EVAL mode.
- In the main() function, create the SMDebug hook and register to the model.
- In the main() function, pass the SMDebug hook to the train() and test() functions in the epoch loop.
- Configure Debugger Rules and Hook Parameters in the main notebook.

Profiling: Using Sagemaker Profiler is similar to using Sagemaker Debugger:
- Create Profiler Rules and Configs
- Pass Profiler Configuration to the Estimator
- Configure hook in the training script

### Results

There are a few errors popping up on the debugging output - please the below list.

- VanishingGradient: I would try to use Xavier initialization to initialize the weights in the neural networks.
- PoorWeightInitialization: I would try using different weight initialization techniques.
- LowGPUUtilization: I would try using GPU.
- ProfilerReport: Not sure what to do with this.


## Model Deployment

To deploy the model in a clean way, I created an inference script (*inference.py*) using PyTorchModel function. 

To query the endpoint, below are the steps -

1. Random pick an image file from the list of folders for different breeds.
2. Resize the image
3. Invoke the endpoint and make prediction. 

