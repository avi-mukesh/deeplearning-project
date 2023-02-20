# Dog Breed Classifier

I am creating a dog breed classifier. The dataset contains images of different dogs. A pretrained ResNet18 classifier is finetuned to this dataset. 

## Project Set Up and Installation

My code is found in the "train_and_deploy.ipynb" notebook.

"hpo.py" is the script used for hyperparameter tuning. Epochs is fixed at 4
"train_model.py" is the script used for training. Epochs is fixed at 10.

For train_model.py script, we require smdebug to be installed inline using PIP, in order for mdoel profiling and debugging to be performed. 

## Dataset

### Overview
The data is a collection of images of dogs of different breeds. There are 133 different dog breeds in each of the training, validation, and test sets. By using the command "find . -type f | wc -l" in the terminal, I discover there are 6680 images in the training set, 835 in the validation set, and 836 in the test set. I obtain the data from this S3 bucket: s3://udacity-aind/dog-project/dogImages.zip.

### Access
To access the data, I use this command "!aws s3 cp s3://udacity-aind/dog-project/dogImages.zip ./" which copies the data locally into the the root directory. I then unzip this data using "!unzip dogImages.zip" before uploading to my S3 bucket using the same "aws s3 cp" command.

## Hyperparameter Tuning
I opted for a pretrained ResNet18 model. I first looked at other image classification models and compared their accuracies and complexities. I found this website https://paperswithcode.com/sota/image-classification-on-imagenet?tag_filter=3 to be quite useful. Resnet-18 achieves an accuracy of 72% on the ImageNet database. This accuracy, for my purposes, is plenty to demonstrate the classifiers ability to be finetuned on a different dataset. Also, Resnet-18 can be accessed from the "torchvision.models" package, which is very convenient. I also thought about using Resnet-50, however, this network is more complex and would require more time to finetune. This would eat up more of my budget, and hence I decided not to use it.

The hyperparameters I decided to change are the training batch size, and the learning rate. The set of hyperparameters I used were as follows
batch size: 32, 64, 128, 256
learning rate: continuous range from 0.001 to 0.1

I fixed the number of epochs to 4 so as to not use too many resources and go over budget. Of course, after selecting the set of hyperparameters that gave the best results, I trained the model again with these hyperparameters, but allowing 10 epochs for better results. I chose to train using a maximum of 5 different set of parameters during the tuning process, as it seems like the tuning algorithm is using an efficient search method to find the best hyperparameters, since the loss was decreasing with each training job.

![Tuning job in progress](/readme-images/hpo-tuning-job.jpg "Tuning job in progress")
![Tuning job completed](/readme-images/hpo-tuning-job-completed.jpg "Tuning job completed")


Following is a screenshot showing the metrics during the training process
![Training metrics](/readme-images/training-metrics.jpg "Metrics")



## Debugging and Profiling
To perform model debugging, I included debugging hook in my training script. I set the hook mode to either "TRAIN" or "VALID" depending on which stage of the training the script was in. I created a list of rules that need to be looked out for. These included vanishing gradients, and the loss not decreasing. I also configured hook parameters so that tensors are saved every 100ms in training, and every 10ms during validation.

### Results
The profiler report shows that the total training took just under 6 minutes. I was quite surprised by this, since during hyperparameter tuning, each training job took about 5 minutes. So having 6 extra epochs didn't make much of a difference. Initially, I though this is likely because I was using GPU instances for training, which results in faster training compared to training using CPU instances.

However, it turns out that the maximum GPU utilisation was only about 50%, CPU utilization peaked at about 97%. I realised that this was because the batch size of only 32 was too small to efficiently use the GPUs. There were, however, only 2 instances when the GPU underutilization rule was triggered. Furthermore, about 250 CPU bottlenecks were found, accounting for 21% of the total time. This is again due to a small batch size.


## Model Deployment
The deployed model takes a torch.Tensor object, and returns an array of 133 numbers. The index with the highest number corresponds to the class that the the model has predicted for this image. So in order to peform inference, the image must first be represented as a torch.Tensor object. The image must also be transformed so it has dimensions 224x224 which is what Resnet-18 requires. Using the folder names, using the os module in Python, I generate a list of the possible class names (dog breeds), and reference this list using this index. The screenshot below shows an example of an inference being made correctly.

![Deployed endpoint](/readme-images/deployed-endpoint.jpg "Deployed endpoint")
![Inference being made](/readme-images/prediction.jpg "Inference being made")