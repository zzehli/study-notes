# Lesson 1 
## lecture
* bird recognizer
* 2012: logistic regression model with data created by experts
* deep learning -> not need to hand code features
    * dl creates complex features (flower, words) in images thru simpler features (color, edges) as the layer adds
* RISE: jupyter notebook for slides, https://rise.readthedocs.io/en/latest/
* In every *epoch*, the training goes through all of the data once
## Book
* deep learning -> neural nets, dated back to 1950s
* fastai paper: fastai: A Layered API for Deep Learning (2020), https://arxiv.org/abs/2002.04688
* machine learning according to Arthur Samuel 
> Suppose we arrange for some automatic means of testing the effectiveness of any current weight assignment in terms of actual performance and provide a mechanism for altering the weight assignment so as to maximize the performance. We need not go into the details of such a procedure to see that it could be made entirely automatic and to see that a machine so programmed would "learn" from its experience.
* neural nets is a particular kind of machine learning model that is highly flexible
    * in NN, the mechanism to update the weight assignment is through stochastic gradient descent
* deep learning terms:
    * the measure of "performance" is called *loss*
    * the *predictions* are calculated from the *independent variable*, which is the data not including the labels
* *loss* and *metric* are different things, loss is used by SGD to update the weights, while metrics are used by human to understand the quality of a model's predictions
* when you a pretrained model, the last layer, which is designed for specific tasks, is removed and replaced with one or more layers of random weights. This last part is called *head*
* use a pretrained model for a tasks different to what it was trained for is called *transfer learning*
# Lesson 2
## Book
* computer's can't generate random numbers, if you feed the same seed, it will generate the same "random numbers" each time
* jupyter notebook: execute `?function_name` or `??function_name` to see docs
* The drivetrain approach: define objective -> levels -> data -> models
* data augmentation with fastai's `aug_transforms()`
* *loss* is a number that is higher if the model is incorrect (especially if it's also confident of its incorrect answer), or if it's correct, but not confident of its correct answer
* clean data by run training on the dataset first and filtering out high loss items and decide whether they are labeled correctly
* a model consists of two parts: architecture and parameters
* problems is in production: out-of-domain data (data that are not seen in training), domain shift (customer use cases change overtime, so new data are different from training data)
* *feedback loop* refers to the model can change the behavior of the system it is part of, such as predictive policing (feedback loop makes negative bias worse)
* recommendation system
> nearly all machine learning approaches have the downside that they only tell you what products a particular user might like, rather than what recommendations would be helpful for a user.
* areas of deep learning application
    * computer vision
    * text (NLP)
    * tabular data
    * recommendation system 
# Lesson 3
## Book Chapter 4
* Tensor
    * shape is the length of each axis
    * the length of a tensor's shape is its rank `x.shape`
    * "dimension" is an ambiguous term, it can mean both "number of axes/rank" or the size of an axis
* task: classify 3 and 7
    * approach 1: get the average of all images of the same digit
* measure the differences between the ideal digit and an instance
    * L1 norm: mean of absolute value of differences
    * L2 norm, root mean squared error (RMSE): the mean of the square of difference and then take the square root
* numpy array is a multidimensional table of data, can be of any data type (all items of the same time), can be jagged array
* tensor cannot be jagged, and more restricted on data types (basic numeric types), but support running on GPUs (Pytorch can also calculate derivatives of operations on tensor)
* broadcasting in pytorch: pytorch will perform operations on two tensor of different shapes by making the tensor with a smaller rank the same size as the larger one
* classify 3 and 7
    * 1st itr: calculate "ideal" pixels for 3 and 7, for each sample, calculate average RMSE between it and ideal 3 and 7s. Whichever got the smaller error will be the result of the classification.
    * add learning (automatically modify itself to improve its performance)
    * 2nd itr: 
        * add weights: optimize a function `prob_eight(x*w)` where x is the image and w is weights; optimize the weights so that the result of the function is high for images of 8 and low otherwise
* the algorithm
```
    Initialize the weights.
    For each image, use these weights to predict whether it appears to be a 3 or a 7.
    Based on these predictions, calculate how good the model is (its loss).
    Calculate the gradient, which measures for each weight, how changing that weight would change the loss
    Step (that is, change) all the weights based on that calculation.
    Go back to the step 2, and repeat the process.
    Iterate until you decide to stop the training process (for instance, because the model is good enough or you don't want to wait any longer).
```
```
for x,y in dl:
    pred = model(x)
    loss = loss_func(pred, y)
    loss.backward()
    parameters -= parameters.grad * lr
```
* *backpropagation/backward pass*: calculate grad
* *forward pass*: calculate activation
* *learning rate* adjust the weights by a small number
    * a step: `w -= gradient(w) * lr` adjust weights in the direction of the slope (increase the weight when slope is negative, because minimum is ahead, vice versa)
* tell the difference between "independent variable" and "weights": independent vars are input of a function, but weights are the parameters
* IMPORTANT example: use SGD to approximate a quadratic function
* IMPORTANT example image classifier:
    * forward pass `def linear1(xb): return xb@weights + bias` where `@` is matrix multiplication in pytorch
    * choose loss function, accuracy won't do, since it is not sensitive to changes in weights
    * `mnist_loss` takes predictions target and output average distance between them after apply sigmoid function to each:
    ```
    def mnist_loss(predictions, targets):
        predictions = predictions.sigmoid() # sigmoid to ensure the predictions are between 0 and 1
        return torch.where(targets==1, 1-predictions, predictions).mean()
    ```
    * *sigmoid* function take in any input and return a value between 1 and 0: `def sigmoid(x): return 1/(1+torch.exp(-x))`
    * *optimization step* update weights based on the gradients; instead of calculating the loss over all dataset, we calculate loss in batch called *mini-batch*
    * Methods in PyTorch whose names end in an underscore modify their objects in place. For instance, bias.zero_() sets all elements of the tensor bias to 0.
* single nonlinearity with two linear layers is enough to approximate any function, but more layers but smaller layers improve more results
```
def simple_net(xb): 
    res = xb@w1 + b1 # linear layer
    res = res.max(tensor(0.0)) # activation function
    res = res@w2 + b2 # linear layer
    return res
```
* Q: Activations: Numbers that are calculated (both by linear and nonlinear layers); how is this related to activation functions?
    * Activation functions introduce non-linearities, allowing neural networks to learn highly complex mappings between inputs and outputs. Without activation functions, neural networks would be restricted to modeling only linear relationships between inputs and outputs. 
    * activations vs. parameters: parameters are randomly initialized then optimized
## Lecture
* estimate a quadratic function
    * create the target function
    * create data points on the model, add noise
    * create loss function (mean square error)
    * add require_grad to the loss function, then can call `loss.backward()` which gives a gradient for each of the parameters 
    * change parameters to by `lr * grad`
* SGD: if gradient is negative, increase the parameter will decrease the loss, therefore we should increase the parameters. Because gradient is negative, we have to subtract negative number (gradient * lr) to increase parameters. (see example of SGD in fastai-example notebook)
* *reactified_linear* (relu) function
```
def reactified_linear(m, b, x): 
    y = m*x + b
    return torch.clip(y, 0) #everything below 0 becomes 0
# or
def simple_net(xb): 
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res
```
* relu is a function that you can add up together to approximate any functions
* The intuition behind relu is "a series of any number of linear layers in a row can be replaced with a single linear layer with a different set of parameters." (ch. 4)
* use excel for SGD
# Lesson 4
## Book Chapter 10
* self-supervised learning use unlabeled data
* to train a classification model, it is useful to fine-tune the language model on unlabeled dataset (movie review in this case) with self-supervised learning, then fine-tune it on the classification data (Universal Language Model Fine-tuning)
* text processing is similar to categorical data (generate a list of vocab, convert to index, combine them into an embedding matrix, the embedding matrix is the first layer of a NN) (see prev chapters)
* *sequence* a list of words in model training
* in training, the independent variables are the sequence of words starting with the first word and ending with the second to last. The dependent variables starts from the 2nd to the last word.
* fine-tune a text generation model: tokenization -> numericalization -> create data loader -> create language model (rnn)
* tokenization: word-based, subword-based and character-based
    * word-based tokenization assumes that spaces is the best unit to separate meaningful components in a sentence, but this might not be the case in many languages, as in Chinese 
    * Size of vocab for subword tokenization: smaller vocab means each token will represent fewer characters (t1 represent `st` as opposed to `street`)
        * smaller vocab size: more tokens to represent a sentence, but faster training (less memory, less things to learn)
        * bigger vocab size: closer to word-level tokenization, less token to represent a sentence, but slower training (more things to learn)
* numericalization process: 1. make vocab list; 2. assign each word with its index
* data loader:
    * at every epoch shuffle the collection of documents and concatenate them into a stream of tokens
    * cut that stream into a batch of fixed-size consecutive mini-streams (create batch)
    * each of the sequence has the same length, with beginning of a sequence denoted by a special token
    * model will then read the mini-streams in order, and thanks to an inner state, it will produce the same activation whatever sequence length we picked
* fine-tuning text generation (movie reviews) 
    * convert word indices into embeddings
    * use RNN to fine-tune text generation movie review
    ```
    learn = language_model_learner(
        dls_lm, AWD_LSTM, drop_mult=0.3, 
        metrics=[accuracy, Perplexity()]).to_fp16()

    ```
    * save the resulting encoder
        * encoder: all but the last layer of the model
    * This RNN perform next-token prediction by default, we need to convert it to a classification model
* fine-tuning classification 
    * sequences also need to have the same length; achieved through padding
    * load the encoder from the previous step
    * fine tuning the model by freeze by the last few layers
# Resources
## 