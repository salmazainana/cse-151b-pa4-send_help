This respository is refered and modified from this repository of the [paper](https://arxiv.org/abs/2109.03079).

To run our code first go to main.py and you will see the baseline train, custom train, and supcon train functions which contain the training loops for our three different models; the last function is what you will run first as it which handles the top level of training for all three models. (predictions, accuracy, batching).

To inspect the inner workings of the model navigate to model.py where you can see the classes for our three models.  We also included our classifier here where we can apply Linear layers and activation functions.

arguments.py contains all of our experiment setup details including hyper-parameters, settings, and different experiment options

loss.py is has a custom loss function for the SuperConLoss.  There is nothing fancy happening with dataloader.py, utils.py, or load.py.
