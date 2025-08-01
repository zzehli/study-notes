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
# Lesson 3 SGD
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
    where
    ```
    w1 = init_params((28*28,30))
    b1 = init_params(30)
    w2 = init_params((30,1))
    b2 = init_params(1)
    ```
    here, 30 is the size of the hidden layer, can be any number
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
    * Q:  w2 are 1s? (because pred = relu1 + relu2, but pred = prelu1 * w2_1 + prelu2 * w2_2)
# Lesson 4 NLP
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
## Lecture
* one advantage of the transformer architecture is to leverage GPU/TPU compute
* lecture based on this [notebook](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners), which compare asks two phrases/words from a patent dataset have the same meaning 
* turn a similarity problem into a classification one ("Does the two phrases the same meaning")
* underfit and overfit
* overfitting is hard to recognize, because it is close to the training data, unlike underfitting
* validation set: hold out from training, a separate dataset to see if the model fits (produce metrics) (huggingface calls this the test set)
* test set: not used for metrics at all, only used in the final assessment of a model
* cross-validation: rarely applicable in real world situations; it only works in the same cases where you can randomly shuffle your data to choose a validation set, but random is often not what you want (think time series)
* Goodhart's Law: when a measure becomes a target, it ceases to be a good measure
* the concept of the outlier only exists in a statistical sense, not in the real world -> don't delete outlier in the real world applications
# Lesson 5 Tabular Data
## Lecture
* build a tabular model from scratch with the titanic dataset, based on this [notebook](https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch); we are making a model to predict if a passenger can survive given information about age, ticket price, etc 
* replace missing values with mode `df.mode().iloc[0]` (choose the first mode if ties are returned)
* EDA with numpy:
    * `df.describe(include=(np.number))`
    * `df.describe(include=[object])`
    * `df['col'].hist()`
* some models don't work well with long-tail distribution
* turn long-tail dist with logs (use logs for things that can grow exponentially like money or population): `df['logCol'] = np.log(df['col'] + 1)`
* turn categorical variables into dummy vars: `pd.get_dummies(df, columns=['col1', 'col2'])`
* [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) is element wise multiplication instead of matrix multiplication
* trailing underscore in pytorch is in-place operation `coeff.requires_grad_()`
* sigmoid function makes sure that the coefficients can be arbitrarily big without worrying about reaching 0 or 1
    * if the dependent var is binary, use sigmoid 
* `sympy` package can plot symbolic functions
* linear model -> nonlinear model (neural net)
* in numpy and pytorch, `+_*/` means element-wise operations, `@` is matrix multiplication
* arbitrary size `n_hidden` for the hidden layer : `layer1 = (torch.rand(n_coeff, n_hidden)-0.5)/n_hidden`
* nn model:
    ```
    def calc_preds(coeffs, indeps):
    l1,l2,const = coeffs
    res = F.relu(indeps@l1)
    res = res@l2 + const
    return torch.sigmoid(res)
    ```
* Q: excel and the nn above: l2 coefficients are not shown in excel; `n_hidden` the col number fo the parameters; relu1 + relu2 are equivalent to `res = F.relu(indeps@l1)` assuming l2 are 1s and const are 0, we will get the `res = res@l2 + const = res`
    * https://chatgpt.com/share/6880f795-a920-8009-82eb-abad1525126a
* turn a NN to a deep NN (deep learning), add more hidden layers (we only have 1 hidden layer in the above example)
    ```
    def calc_preds(coeffs, indeps):
    layers,consts = coeffs
    n = len(layers)
    res = indeps
    for i,l in enumerate(layers):
        res = res@l + consts[i]
        if i!=n-1: res = F.relu(res)
    return torch.sigmoid(res)
    ```
    where `relu` and `sigmoid` are activation functions: sigmoid for the last layer, relu otherwise
* for tabular data, mindlessly apply deep learning doesn't really help; feature engineering is needed
* *ensemble*: combine multiple models (eg: create 5 models and average the results)
* [random forest](https://www.kaggle.com/code/jhoward/how-random-forests-really-work/)
# Lesson 6 Random Forest
* continuation of random forest and tabular data chapter on the book
* oneR classifier use binary splits to arrive at an optimal prediction model based on the best binary splits
* expand oneR to a decision tree by splitting the classifier further decision points based on other parameters
* use `sklearn` to create a decision tree
* *gini*: draw a sample, how likely is it to draw two of the same thing consecutively
* use decision tree for tabular data as a baseline
* decision trees don't care the order of the data or the distribution of the data
* improve the results: *bagging* generate a bunch of unbiased and uncorrelated models and take the avg of their descriptions
* to perform bagging on decision trees, create decision trees on subsets of the dataset, this give us *random forest* 
* *feature importance plot* is a plot in random forest that tells how important an independent var is  (by adding up the improvement on gini in RF)
    * a general tip to get a sense of which var is more important
* the more trees in the RF, the lower the error (decreasing in returns)
* out of bag (oob) error: another way to calculate error without a validation set by validate the model on rows not included in the training
* Jeremy is a RF guy
* how RF help with model prediction? 
    * How confident are we in our predictions using a particular row of data?
    * For predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?
    * Which columns are the strongest predictors, which can we ignore?
    * Which columns are effectively redundant with each other, for purposes of prediction?
    * How do predictions vary, as we vary these columns?
* *partial dependence plot* is not restricted to RF; it is a more general concept 
    * avoid cross correlation
    * something is caused by x and not y
* *gradient boosting* use smaller tree, then use the residual to create another tree, etc; the prediction is the sum of the predictions (not average)
    * generally more accurate than RF
    * can overfit, unlike RF
* Paddy Rice dataset kaggle demo
    * set random seed for reproducibility
    * preprocess data/images
    * quickly iterate: how to try out many many models instead of stuck in 1 model; try to train a model in 1 min
    * PIL images are w x l instead of l x w
    * best vision model for fine-tuning in PIL library: https://www.kaggle.com/code/jhoward/the-best-vision-models-for-fine-tuning
    * when you use a model, it is not necessary to learn everything about how the model works
    * create new notebooks as you iterate
    * JH doesn't do hyperparameter search, doesn't use autoML
    * CV -> refers to the table above; tabular: RF, then GBM (does do hyperparameter search with GBM)
    * more competition techniques https://www.kaggle.com/code/jhoward/small-models-road-to-the-top-part-2/ 
    * image resize methods (crop, pixel, etc), model selection, test time augmentation
    * can only use tta when there are predefined augmentations during training (tta only makes it easy to apply during test time)
# Lesson 7 Collaborative Filtering
## Book Chpater 8
* collaborative filtering deal with recommendations
* latent factors: there are underlying concepts about people's preference 
* We have a table of users' rating of movies with lots of missing ratings that a user hasn't watched. The goal is to guess what these ratings would be
* dot product-based model training process
    * for each of the user, and movieId, we use 5 latent factors and initialize them as random parameters
        * we show a dot product of movie x and user y as a table
    * to calculate the predictions, take the dot product
        * each latent factor represent something meaningful: a user likes an action movie or a movie is action-heavy, if that is the case, the dot product will be high, otherwise low
    * to calculate the loss, we pick mean square error
    * use SGD to train the model 
* how to perform vector index? multiple one-hot encoding matrix with the target vector (matrix) of movie/user parameters
* use embedding layer to accelerate the matrix multiplication above
* if we only have weights, than we can't say things like "this movie is more popular than other movies" or "this movie is better than others", we need to introduce biases
```
class DotProductBias(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.movie_bias = Embedding(n_movies, 1)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        res = (users * movies).sum(dim=1, keepdim=True)
        res += self.user_bias(x[:,0]) + self.movie_bias(x[:,1])
        return sigmoid_range(res, *self.y_range)
```
* weight decay, or L2 regularization, consists in adding to your loss function the sum of all the weights squared; it prevents over-fitting
    * for a parabola: y = ax^2, weight decay is applied to loss with:
    $$ loss_{total} = loss_{orig} + wd \cdot a^2$$
    which gives us the gradient:
    $$grad_{total} = grad_{orig} + 2 a \cdot wd $$
    
    we have `a = 5, lr = 0.1, wd = 0.01 and grad_from_data = 4`:
    $$ grad_{wd} = 2 \cdot wd \cdot a =  2 \cdot 0.01 \cdot 5 = 0.1 $$ 
    $$ grad_{total} = grad_{data} + grad_{wd}  = 4 + 0.1 = 4.1 $$
    update the param $a$:
    $$a = a - lr \cdot grad_{total} = 5 - 0.1 \cdot 4.1 = 4.59 $$
    compared to
    $$a = a - lr \cdot grad_{data} = 5 - 0.1 \cdot 4 = 4.6 $$
    weight decay makes the parameter closer to 0. In this case $a$ is the weight, we want the weight to be smaller to prevent big changes.
* in Pytorch, use `nn.Parameter` to mark params as trainable 
* use principle component analysis to analyze embeddings
* embedding distance: we can use distance between two items to define similarity: rank this distance gives items that are most/least similar to the given item
* *bootstrap*: how do you train when you have a limited amount of data? One strategy is to use a tabular model based on user meta data to construct your initial embedding vector
* a small amount of super users can bias your recommendations: anime users that watch and rate lots of anime; for latent factors, it is hard to detect the biases
    * In a self-reinforcing system like this, we should probably expect these kinds of feedback loops to be the norm, not the exception
* deep learning-based model: instead of calculating the dot product of embeddings, concatenate them and put them in a neural net
```
class CollabNN(Module):
    def __init__(self, user_sz, item_sz, y_range=(0,5.5), n_act=100):
        self.user_factors = Embedding(*user_sz)
        self.item_factors = Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(user_sz[1]+item_sz[1], n_act),
            nn.ReLU(),
            nn.Linear(n_act, 1))
        self.y_range = y_range
        
    def forward(self, x):
        embs = self.user_factors(x[:,0]),self.item_factors(x[:,1])
        x = self.layers(torch.cat(embs, dim=1))
        return sigmoid_range(x, *self.y_range)
```
* `**kwargs` in a parameter list means "put any additional keyword arguments into a dict called kwargs
## Lecture
* kaggle competition
    * iterate quickly
    * train with smaller dataset to see how much memory is needed
    * add fastai `GradientAccumulation` callback and parameter `accum` to reduce memory usage: your batch size becomes `bs=num//accum`, meanwhile update the weights every `accum` number of iterations
* Road to Top kaggle notebook
    * fastai's `DataBlock`
    * in fastai, loss function is automatically selected by the learner
    * in general, the first and the last layer in the model are very important, including the loss
    * softmax $$\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}$$
    * cross-entropy loss: based on softmax, see https://chris-said.io/2020/12/26/two-things-that-confused-me-about-cross-entropy/ for further infomation
* collaborative filtering: book chapter example
    * initialize random weights
    * use SGD and a RMSE to improve the weights
    * the core of collaborative filtering is matrix completion: complete the missing values in the matrix
    * embedding
    * how to characterize user's personal preferences: some tend to give high scores to all movies while others don't give high scores? Through biases (1 col for user bias and movie bias respectively)
* weight decay (L2 regularization): add to the loss function the sum of all weights squared
    * intuition: the sum of all weights squared is a small number that is less than 1. This way, we make the loss function go down (less steep). The bigger the weight decay, the less the weights go down
## Concepts
* latent factors
* weight decay, L2 regularization
* embeddings
* cross-entropy loss
# Resources
* need to read nlp deep dive chapter
* foundation chapter (17)
* computational linear algebra short course
# Course Review
Part I of the course focuses on four core aspects of deep learning: vision, NLP, tabular data and collaborative filtering (recommendation). The course starts with a gentle introduction to modern DL with the `fastai` library. It then dives into the building blocks of neural network in Lesson 3 with Stochastic Gradient Descent and basic neural network. This is a highlight of the course, especially the book, which starts from non-NN example of SGD and then proceed to introduce SGD with NN. This approach illustrates the fact that many parts of NN/DL algorithm are swappable. For example, the forward function can be an NN, but it can also be an average, a parabola or a dot product. Throughout the four core areas, the same group of basic concepts are applied repeatedly, which gives a deeper understanding of these concepts.

The concepts are taught though concrete examples, often a particular dataset. These examples are largely informed by JH's experience with Kaggle competition, which though out the course, he gives many tips on how to approach these competitions. Overall, this is an excellent first introductory course to anything ML.One drawback of this course is its reliance on the fastai library, which dramatically simplifies the ML operations but also introduces abstractions of its own. To my knowledge, the fastai libraries are not very popular nowadays beyond the fastai community.

JH has a philosophy about how to best teach ML to beginners that do no rely on math. This is much appreciated and rare in this field. The fastai community is a strong and supportive community to get involved in ML/AI. Many alumni of the course has gone on to become prominent practitioners in the field.