# [Intro to Large Language Models, Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g)
* LLM in practice consists of two files, parameters file and run file
* getting the parameters is the hard part:  model training
* running the model on your laptop is easy: model inference
* model training (pretraining stage, stage 1):
    * think of it as compression of a chunk of internet
    * train with GPU, expensive: train llma 2 70b cost $2m, can take around 12 days
    * 70b parameters (140g in size) is rudimentary level model (chatGPT would x10)
    * this is the base model
* neural network: predict the next word in the sequence
    * given 4 words: "cat" "sat" "on" "a", the model would predict the next word with high probability ("mat" with 97%)
    * to predict accurately, the model has to learn about the world, and then compress the information into the parameters
    * in the inference stage: the networks "dreams" internet documents, a process called hallucination (a combination of facts and things that are made up)
* how does NN work?
    * architecture like Transformer is very clear
    * however, the how parameters are distributed within the network is unclear
    * we can only measure that this works
    * think of them as inscrutable artifacts
* Stage 2: finetuning -> assistant model (analogy: from document generator to assistant that can answer questions)(https://youtu.be/zjkBMFhNj_g?si=ydzVgqC89H5QuYY3&t=1081)
    * task is the same, predictions
    * dataset is different, from internet documents to conversations produced by real people (write labeling instruction -> hire people to produce high quality Q&A responses)
    * optional labeling method: comparison
    * finetune base model based on this data, typically takes 1 day
    * this stage is about alignment, changing the formatting from internet documents to helpful assistant
    * obtain assistant model
* LLM Scaling: performance of LLM is a smooth, well-behaved function of 
    * N, the number of parameters
    * D, the amount of training text
* 28:00 LLM can use tools to help it generate an answer: browser, calculator, generate graph with python, generate image with Dall-E
* multimodality: 
    * generate and see pictures
    * talk to models
* future directions
    * think fast and slow: currently, LLM can only think in fast mode
        * what if the model can take time to come up with a better result?
    * self-improvement (e.g. AlphaGO)
        * AlphaGo is not LLM, it learns by imitate human players and by play against itself
        * can LLM self-improve like AlphaGo?
        * lack of reward function to tell LLM if an answer is good or bad
    * Custom LLMs
    * LLM as OS: LLM at the core of an OS
* LLM security
    * jailbreak (trick LLM to produce things it does not supposed to)
    * prompt injection (inject additional prompt to LLM)
    * data poisoning (malicious data used in training model can poison LLM)
# Neural Network, 3Blue1Brown
## [Chapter 1. neural network (multilayer perceptron)](https://youtu.be/aircAruvnKk?si=BBdfFO5u6oVUcr3c)
* neural network as a metaphor (example, recognize an image of digits):
* neuron: a thing that holds a number (the first layer in the neural network is the all pixels of the image, the last is the probability of the possible results); the values (normalized between 0 to 1) of neurons are called *activation*; neurons are also functions that take output from the previous layer and output an activation
* activation in one layer determines the activation in another layer, which is similar to how neuron works
* the middle layers are called hidden layer
* the goal is to have a neuron recognize one aspect of the image, eg. a horizontal edge
* from one layer to another, calculates the weighted sum of the activation, which also include a bias and a normalization function (sigmoid)
* *learning* refers to the process of finding the right weights and biases
## [Chapter 2. Gradient descent, NN training](https://youtu.be/IHZwWFHWa-w?si=h6RbFvXM-zv9QmnF)
* start the training with random weights and bias, calculate the *cost*; the cost is small when the prediction is correct and large when wrong
* the cost meansures the performance of the parameters
* to tell the model to perform better, minimize the cost function by telling the model where to move (towards a local minimum) based on the direction of the fastest descent, aka the *gradient* of the location
* the algorithm for computing the gradient efficiently is called *backpropagation*
## [Chapter 3. Backpropagation](https://youtu.be/Ilg3gGewQ5U?si=Y1ge6ogfs8sFxBra)
* adjust weights, bias so that desired activation of neurons are achieved
* use gradient descent to adjust weights and bias; use backpropagation to calculate gradient
* each neuron is connected to every neuron in the next layer, each of these neurons require different adjustments to this neuron, summing them get us the desired change, propagate this process backwards becomes the intuition of backpropagation
* because backpropagation on each neuron is costly, we batch training set and perform the backpropagation on a batch; this is faster; this is called *stochastic gradient descent*
## [Chapter 4. Backpropagation calculus](https://youtu.be/tIeHLnjs5U8?si=828RZzkAjBPHsiMO)
* use chain rule to backpropagate (cost as a function of previous layers' weights * activate - bias) how each weights and bias affect the cost function to achieve gradient descent
## [Chapter 5. How LLM work](https://youtu.be/wjZofJX0v4M?si=-tTdjMPGNNMZ5ckH)
* GPT: generative pretrained transformers
* transformers are mechanisms for next-word-predictions
    * GPT generate one word at a time based on all the words that's given and generated by itself
* structure of transformers
    * sentence into token (roughly equals to individual words)
    * token into vectors, words with similar meaning are placed close-by
    * back and forth, matrix multiplication
        * vectors go thru attention block, where they interact with each other
            * values in vectors are updated, update meaning based on context in the sentence
        * then, vectors go through the next stage, multipayer perceptron
            * ask questions to each vector and update the values
    * the last vector will capture the meaning, which produce a probability distribution of possible tokens that will come next
* deep learning is a class of model that scales well with large data set
    * deep learning include MLP, CNN and transformers
    * they uses backpropagation as training algorithm
    * backpropagation algorithms are formated similarly, where input are higher dimension array (*tensor*), which is then transformed by another matrix called *weights*
    * weights can be through of as matrices, and input data as arrays
* gpt-3 weights are organized into matrices of 8 categories: embedding, key, query, value, output, up-projection, down-projection, unembedding
* the model has a list of words, each associated with a vector, together, they form *embedding matrix* (E)
    * word embedding (word vector) in gpt-3 has 12,288 dimensions, with 50k words
    * each direction in the space tends to have a meaning: eg: E(Mother) - E(Father) = E(woman) - E(man)
    * in the inference stage, the initial embedding from the embedding matrix gets more context thru other input words, altering the vector values; in gpt-3 the context length is 2048 words
* last step, come up with a possible list of word predictions
    * use the last vector in the embeddings, times a matrix, called *unembedding matrix*, to get a vector of 50k values (each word in the gpt-3 model gets a value); this umembedding matrix has 50k row and 12288 dimension
    * turn this vector into a probability distribution with a normalization function called softmax
## [Chapter 6. Attention in transformers](https://youtu.be/eMlx5fFNoYc?si=k8-PyFulowhBQ1-T)
* the notion of transformer is to change word embeddings to incorporate contextual information
* what does attention (layer/block) do?
    * move a generic embedding saved as part of the embedding matrix to a more contextual specific direction
    * single head of attention (self-attention) (query, key and value matrices; query and key matrices are each 12288 cols and 128 rows, value matrix contains the same parameter as the sum of the two)
        * produce query vector as questions to solicite more context for each token
        * produce key vectors as answers to the questions 
        * matching query and key vectors will points to the same direction, therefore, the dot products of the query and key vectors are large positive numbers
        * *attention pattern* is the dot product of key vectors and query vectors to assess the relevance of embeddings, the bigger the dot product, the more relevant they are (reference the attention is all you need paper)
        $$softmax({QK^T}/{\sqrt(d_k)})V$$
        * the size in the attention pattern is the size of the context window
        * attention pattern allows the model to answer: which words in the context are relevant to which other words
        * value matrix (the third matrix) generate a vector that moves the embedding into the desired direction (how to move "creature" to "fluffy"?)
        * value matrix formation contains a low rank transformation to reduce the parameter size
    * In one layer, query, key and value matrices together contains the same amount of parameters as embedding/unembedding matrix. This is single headed attention, but GPT has 96 attention heads, which makes attention layer 96 times the size of a single attention head. The attention layer accounts for 1/3 of total parameters
    * full attention are multi-headed attention: GPT has 96 attention heads
    * cross-attention involve two types of data, such as audio-text or translation; one languages are self-attention
# [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0?si=pJ88FgwuDpLq2OgR)
# [Building Large Language Models (LLMs), Yann Dubois](https://www.youtube.com/watch?v=9vM4p9NN0Ts)
# Micrograd, Andrej Karpathy
* Backpropagation is the recursive application of chain rule backward thru the computational graph
* a mathematical model of neuron, artificial neuron: $$f(\sum_i w_ix_i + b)$$ where $w_ix_i$ is weighted input and $b$ is bias, and $f$ is an activation function (sigmoid, ReLU) that normalizes the weighted output
# Math topic
## Low rank transformation
# Curious topics
## A Mathematical Framework for Transformer Circuits (reverse engineer attention models)
