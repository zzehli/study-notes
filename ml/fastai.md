# Lesson 1 
## lecture
* bird recognizer
* 2012: logistic regression model with data created by experts
* deep learning -> not need to hand code features
    * dl creates complex features (flower, words) in images thru simpler features (color, edges) as the layer adds
* RISE: jupyter notebook for slides, https://rise.readthedocs.io/en/latest/
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
# Resources
## 