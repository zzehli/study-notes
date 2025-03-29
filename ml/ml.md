# Models
## [Intro to Large Language Models, Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g)
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
## Neural Network, 3Blue1Brown
### [Chapter 1. neural network (multilayer perceptron)](https://youtu.be/aircAruvnKk?si=BBdfFO5u6oVUcr3c)
* neural network as a metaphor (example, recognize an image of digits):
* neuron: a thing that holds a number (the first layer in the neural network is the all pixels of the image, the last is the probability of the possible results); the values (normalized between 0 to 1) of neurons are called *activation*; neurons are also functions that take output from the previous layer and output an activation
* activation in one layer determines the activation in another layer, which is similar to how neuron works
* the middle layers are called hidden layer
* the goal is to have a neuron recognize one aspect of the image, eg. a horizontal edge
* from one layer to another, calculates the weighted sum of the activation, which also include a bias and a normalization function (sigmoid)
* *learning* refers to the process of finding the right weights and biases
### [Chapter 2. Gradient descent, NN training](https://youtu.be/IHZwWFHWa-w?si=h6RbFvXM-zv9QmnF)
* start the training with random weights and bias, calculate the *cost*; the cost is small when the prediction is correct and large when wrong
* the cost measures the performance of the parameters
* to tell the model to perform better, minimize the cost function by telling the model where to move (towards a local minimum) based on the direction of the fastest descent, aka the *gradient* of the location
* the algorithm for computing the gradient efficiently is called *backpropagation*
### [Chapter 3. Backpropagation](https://youtu.be/Ilg3gGewQ5U?si=Y1ge6ogfs8sFxBra)
* adjust weights, bias so that desired activation of neurons are achieved
* use gradient descent to adjust weights and bias; use backpropagation to calculate gradient
* each neuron is connected to every neuron in the next layer, each of these neurons require different adjustments to this neuron, summing them get us the desired change, propagate this process backwards becomes the intuition of backpropagation
* because backpropagation on each neuron is costly, we batch training set and perform the backpropagation on a batch; this is faster; this is called *stochastic gradient descent*
### [Chapter 4. Backpropagation calculus](https://youtu.be/tIeHLnjs5U8?si=828RZzkAjBPHsiMO)
* use chain rule to backpropagate (cost as a function of previous layers' weights * activate - bias) how each weights and bias affect the cost function to achieve gradient descent
### [Chapter 5. How LLM work](https://youtu.be/wjZofJX0v4M?si=-tTdjMPGNNMZ5ckH)
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
### [Chapter 6. Attention in transformers](https://youtu.be/eMlx5fFNoYc?si=k8-PyFulowhBQ1-T)
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
## [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0?si=pJ88FgwuDpLq2OgR)
* Backpropagation is the recursive application of chain rule backward thru the computational graph
* a mathematical model of neuron, artificial neuron: $$f(\sum_i w_ix_i + b)$$ where $w_ix_i$ is weighted input and $b$ is bias, and $f$ is an activation function (sigmoid, ReLU) that normalizes the weighted output
# Prompt Engineering
## [Prompt Engineering Overview](https://youtu.be/dOxUroR57xs?si=YDSjolN3mo3FzvHG)
* Prompts involve instructions and context passed to a language model to achieve a desired task
* prompt consists of
    * instruction
    * context
    * input
    * output
* `temperature` and `top_p` controls how deterministic the model is, 
    * low for exact answers, high for diverse response
* common tasks
    * text summarization
    * question answering
    * text classification
    * role play
    * code generation
    * reasoning
* techniques
    * few-shot prompts: provide examples for how to respond
        * this is useful because the model can do in-context learning, without changing parameters
    * chain-of-thought (CoT) prompting: provide some of the answers to questions that requires reasoning
    * zero-shot cot:  add "let's think step by step" to the original prompt
    * self-consistency: provide some examples for the model, the examples involve 
    demostrating the reasoning for the model
    * generate knowledge prompting: make the model to generate some "knowledge" first
    * program-aided language model (PAL): use programs to read and generate intermediate steps to lead to answer that requires reasoning
    * ReAct: LLMs generate both reasoning traces and task-specific actions
        * it allows LLMs to interact with external tools
* tools and applications
    * LLMs and external tools
        * an *agent* powered by LLM to determine which action to take
        * a *tool* used by the agent to interact the external world
    * Data-augmented generation (DAG not RAG)
        * incorporating external data
    * model safety: increase the safety of LLMs thru prompt engineering
    * RLHF (reinforcement learning from human feedback): train LLMs to fit human preference
        * collect high-quality prompt datasets

## [Prompt Engineer Guide](https://www.promptingguide.ai/)

# Applications
## AI Engineering, Chip Huyen
### Chapter 1, Intro
#### Language Models
* ML models vs foundation models
* token (words or part of a word) -> vocabulary (100, 000 for GPT-4)
* there are fewer unique tokens than words
* two types of language models: masked language model (predict based on before and after the targeted token) and *autoregressive language model* (based on before the toke, more popular now)
* language models are capable of *self-supervision*
* supervision refers to the use of labeled data, while self-supervision models can infer labels from input data
#### Foundation Models
* *Foundation model* refers to the fact that models can handle different data modalities, in addition to text
* They are capable of a wide range of tasks, such as sentiment analysis or translation
* 3 common techniques to adapt a model to specific tasks: prompt engineering, RAG and finetuning
p28
#### ML Engineer vs AI Engineer
* ML engineers develop ml models, which are often smaller and less resource-intensive whereas AI engineers use existing models that are more expensive to develop in-house
* ML engineers focus more on modeling and training and AI engineers focus more on adaptation
* model adaptation techniques have two categories: prompt engineering and finetuning; the latter requires updating the model weights but see significant improvement on model performance
#### Model development 
* three parts modeling and training, dataset engineering and inference optimization
* training
    * pre-training involves training a model for text-completion, this is often the most resource-intensive step in training
    * finetuning and post-training are used interchangeably, they might be done by model developers or application developers
* dataset engineering
    * for ML engineers, this means curate, generate and annotate the data needed for training AI models
    * foundation models have open-ended outputs and take in unstructured data, which requires higher quality datasets
    * for AI engineers, this means deduplication, tokenization, context retrieval, and quality control
#### Application development
* evaluation
    * the open-ended nature make eval important, since there is no standardized outcome
* prompt engineering
* AI interface
    * ai applications can be standalone products instead of becoming part of a product (recommender systems, fraud detection)
        * Streamlit, Gradio, Plotly Dispatch
* ML engineers start with training a model while ai engineers build products first before thinking about models (Shawn Wang, [The Rise of AI Engineer](https://www.latent.space/p/ai-engineer))
### Chapter 2, Foundation Models
#### Training Data
* web data from Common Crawl
* many languages are absent in training data, which lead to worse performance for the models with these languages
* some languages requires more token to represent, thus training them are more expensive
* domain specific tasks are harder for general purpose models to perform since data might not be available on the public internet
#### modeling
* transformer architecture is the most popular one
* transformer is based on seq2seq
* inference for transformer-based models consist of 2 steps
    * prefill: process input tokens in parallel, creating intermediate states that has key value vectors for all input tokens
    * decode: generate one output token at a time
* The attention mechanism computes how much attention to give an input token by performing a dot product between the query vector and its key vector. A high score means that the model will use more of that page’s content (its value vector) when generating the book’s summary
* Other architectures: AlexNet (2012), seq2seq (2014-18), GANs (2014-19), transformer (2017-)
* model size: if a parameter is stored using 2 bytes, a 7B model requires 7 * 2 = 14B bytes (14GB) to run inference
* sparse model requires less resource to run
* models are trained on datasets that have trillions of tokens
* model pre-training compute requirement is measured in FLOP (floating point operations)
* Given a compute budget, the rule that helps calculate the optimal model size and
dataset size is called the Chinchilla *scaling law*
    * They found that for compute-optimal training, you need the number of training tokens to be approximately 20 times the model size
* for smaller models, hyperparameter can be adjusted, but training large models are expensive therefore changing hyperparameters are not possible
    * scaling extrapolation is a research subfield that tries to predict, for large models what hyperparameters will give the best Performance
* scaling bottlenecks: training data and electricity
    * internet data is running out
    * the impact of AI-generated data on models
    * data centers are estimated to consume 1–2% of global electricity. This number is estimated to reach between 4% and 20% by 2030
#### Post-Training
* why is post-training important?
    * foundation models that pre-trained with self-supervision is optimized for text completion, not conversation
    * address abusive language that might arise from the model
* process: supervised finetuning (SFT) and preference finetuning
    * SFT: use high-quality instruction data to optimize models for conversation
    * preference finetuning: typically done with reinforcement learning to align with human preference
* think of post-training as unlocking the capabilities that the pre-trained model already has but are hard for users to access via prompting alone
* supervised finetuning leverages demonstration data that shows the model how to behave; it follows the format prompt + response
* demonstration data are created by human labelers, a pair of prompt and response takes up to 30min to generate, can cost around $20
* RLHF
    * train a reward model that scores models' output
    * optimize the model to generate response that maximize scores
* to train a reward model, labelers compare two responses and choose the better one
##### sampling
* sample next token
    * models create output based on probabilities of possible outcomes
    * greedy sampling is a strategy that always picks the candidate with the highest probabilities, this works fine in classification, but it doesn't work well in generation
    * to calculate the probability, apply softmax to a logit vector, where each logit corresponds to a token
    * sample with a *temperature*: higher temperature makes the rare tokens more likely to appear, thus make the responses more creative 
        * logit gets divided by temperature before applying softmax
        * temperatures are between 0 and 2
        * 0.7 is recommended for creative use
    * logprobs are probabilities in the log scale: it's helpful to evaluating models
    * Top-K: a sampling strategy to reduce computational workload by picking top-k logits to calculate softmax (instead of all logits)
        * a smaller K means a smaller set of choices, thus making the model less creative
    * Top-P: the model sums the probabilities of the most likely tokens in descending order and stops when the sum reaches p (.9 to .95). Only values within this range are considered
        * work well in practice, not in theory
    * stopping condition: let the model know when to stop generating text
        * stop tokens: model stops when it encounter stop tokens
* sample the whole output
    * test time compute: generate multiple responses and select from them (allocating more compute to generate more outputs during inference)
    * best of N: choose one that works the best
    * beam search: expand on the most promising response at each step
    * to pick the best response, sum up logprobs of all tokens in the text, an divided by its length, and pick the highest one
    * can also use human and reward models to pick
* A model is considered robust if it doesn’t dramatically change its outputs with small
variations in the input. 
* structured outputs
    * easy solutions
        * prompting
        * post-processing: eg. write scripts to correct repeated errors from models
    * constrained sampling: filter logits that do not meet the constraints, such as grammar rules
        * establishing such rules are expensive
    * finetuning
        * *feature-based transfer* adding a classifier layer to the model to guarantee the output format
    * the way models sample their responses make them *probabilistic*
        * *inconsistency* refers to model generating different responses for the same or slightly different prompts
        * *hallucination* refers to model giving a response that isn't grounded in facts, unclear why
            * hypothesis 1: a model can’t differentiate between the data it’s given and the data it generates
            * hypothesis 2: hallucination is caused by the mismatch between the model’s internal knowledge and the labeler’s internal knowledge. 
### Chapter 3, Evaluate Foundation Models
#### model eval metrics
* language model metrics
    * Claude Shannon's 1951 paper "Prediction and Entropy of Printed English" used *cross entropy* and *perplexity* as measures
    * four main metrics: cross entropy, perplexity, bits-per-character (BPC), bits-per-byte (BPB) 
* when training a model, the goal is to get the model to learn the distribution of the training data
* entropy measures how much information a token carries (how difficult to predict what comes next)
    * the lower the entropy, the more predictable the model is
* cross entropy measures how much info a model carries, measured by two things
    * training data's predictability, its entropy: $H(P)$ where $H$ is (cross) entropy and $P$ is the true distribution
    * how the distribution by the model diverges from true distribution of the training data
        * measured by Kullback-Leibler (KL) divergence: $D_{KL}(P| |Q)$
    * a model's cross entropy is the sum of the two $H(P, Q) = H(P) + D_{KL}(P| |Q)$
* if a model learns perfectionly, the KL divergence is 0
* bits is a unit of entropy and cross entropy
* if a model's cross entropy is 6 bits, it needs 6 bits to represent token
* since models' tokenization methods differ, bits-per-character would make more sense:
    * If the number of bits per token is 6 (its cross entropy) and on average, each token consists of 2 characters, the BPC is 6/2 = 3
* since # of bits used for a character can differ in an encoding method, *bits-per-byte* becomes a more standardized measure
    * the number of bits a language model needs to represent one byte of the original training data
    * If the BPC is 3 and each character is 7 bits, or 7/8 of a byte, then the BPB is 3 / (7/8) = 3.43.
    * Cross entropy tells us how efficient a language model will be at compressing text. If the BPB of a language model is 3.43, meaning it can represent each original byte (8 bits) using 3.43 bits
* perplexity is the exponential of entropy and cross entropy
    * if a dset with true distribution of $P$, perplexity or PPL is defined as $PPL (P) = 2^{H(P)}$
    * the PPL of a language model on this dset is $PPL (P, Q) = 2^{H(P, Q)}$
    * base 2 is the unit of bit that is used in entropy, but it can be switched to $nat$ as well, which is base e
    * perplexity measures the amount of uncertainty it has when predicting the next token. Higher uncertainty means there are more possible options for the next token
    * If all tokens in a hypothetical language have an equal chance of happening, a perplexity of 3 means that this model has a 1 in 3 chance of predicting the next token correctly
* The more accurately a model can predict a text, the lower these metrics are
    * More structured data gives lower expected perplexity
    * The bigger the vocabulary, the higher the perplexity
    * The longer the context length, the lower the perplexity
* perplexity is lower on data it has seen and higher on unpredictable data
#### Exact evaluation
* evaluation can be exact and subjective; exact eval leaves no ambiguities
* for open ended tasks, exact eval methods include functional correctness and similarity measurements
* functional correctness
    * measures whether the model performs the tasks it is assigned
    * code gen tasks can be measured by executing automated unit tests
* similarity measurements: measure model response with reference data generated by human or AI
    * to compare similarity, metrics include exact match (0 0r 1), lexical similarity (sliding scale) and semantic similarity (sliding scale)
    * lexical similarity:
        * approximate string match
        * n-gram similarity
    * semantic similarity:
        * transform text into a numerical representation, embedding, and compare cosine similarities of two embeddings
        * whether this works well depend on how good the embedding algorithms are
* embedding
    * an embedding algorithm is good if more-similar texts have closer embeddings
#### subjective evaluation
* AI as judge: most common eval method now
    * judge criteria (prompt to AI) isn't standardized
    * increased costs and latency due to additional inference cost
    * self-bias: model favors response generated by itself
    * verbosity bias: favors verbose responses
    * considering cost and latency, one might use cheaper models as judges
    * smaller models might make sense in specialized tasks
    * self-evaluation can be used as sanity checks
* ranking models with comparative evaluation
    * pair-wise comparison is usually easier to conduct than absolute scores for open ended answers
    * the challenge is to determine what question can be address with comparison and what can't
    * ranking algorithms will calculate ranking for each models in a group of pair-wise comparisons
    * resource intensive comparing models require many rounds of comparisons
    * transitivity in model matches might not work in model comparison (A is better than B, B is better than C, but A is not necessarily better than C)
### Chapter 4, Evaluate AI System
* given a criteria (latency, domain specific, etc), how to choose eval metrics?
* domain specific capacities are often evaluated with exact evaluation
    * coding tasks can be evaluated by running it: functional correctness
    * non-coding tasks are evaluated with close-ended tasks, for example, let the model to do multiple choice questions
    * MCQs are not ideal for evaluating summarization, translation and writing
* generation capacity: factual consistency and safety
    * factual consistency
        * measure hallucination
        * use local context to store facts or use a global knowledge source
        * the hardest part is to determine what the facts are
        * AI as a judge can be used to evaluate factual consistency
        * related to the study of *textual entailment* in NLP, which measures the relationship between two statements: entailment, contradiction and neutral
            * can use specialized models for this
        * important for RAG, since the generated response should be consistent with RAG
    * safety
        * inappropriate language
        * harmful recommendations
        * hate speech
        * violence
        * stereotypes
        * bias
* instruction-following capacity: essential for structured output
* role-playing
    * RoleLLM benchmark
    * hard to automate
* cost and latency
* model selection workflow
    * hard attributes: model size, privacy, etc
    * soft attributes: accuracy, factual consistency
    * data contamination: models is tested on the data it is trained on
* design eval pipeline
    1) evaluate all components in a systems: both per task and end-to-end
        * turn-based: evaluate the qualify of each output
        * task-based: whether the system completes a task
    2) create an evaluation guideline
        * define evaluation criteria eg. relevance, factual consistency, safety
        * create score rubrics
        * tie evaluation metrics to business metrics: eg. factual consistency of 80%: we can automate 30% of customer support requests
    3) define evaluation methods and data
        * When logprobs are available, use them. Logprobs can be used to measure how confident a model is about a generated token
        * annotate eval data, which can be reused to create instruction data
### Chapter 5, Prompt Engineering
* in-context learning: teach a model to learn through examples in the prompt
    * each example is called a *shot*, learning from examples are called *few-shot* learning 
* system prompt and user prompt
    * system prompt and user prompt are combined into the input of the model
    * models are trained to prioritize system prompts
* prompts and context lengths are different things (unclear)
    * prompts are input into the model and context is the information available to the model
    * context length is crucial 
* best practices
    * clear and explicit instructions
        * ask the prompt to adopt a persona
        * provide examples
        * specify output format
    * provide sufficient context through context construction (RAG, web search)
        * sometimes it's desirable to restrict model's knowledge to its context (npc in games); but this is not guaranteed, pretraining data might leak into responses
    * break down complex tasks to subtasks 
        * support bot: 1. intent classification 2. generate response based on intent
        * prompt decomposition has tradeoffs (observability, concurrency vs cost)
    * chain of thought (CoT): ask a model to think step by step
        * first prompt technique to works well across models
        * self-critique asks a model to evaluate its own response
    * iterate, version your prompts
    * prompt engineer tools: Open Prompt, DSPy, TextGrad
        * some use AI to generate prompts
        * additional cost due to hidden model calls
    * organize and version prompts
        * put prompts in a separate file, wrap them in a class object to store metadata
        * consider separate codebase with prompts, since prompts might not update with the code at the same pace
* security: 
    * prompt extraction: get prompts used by the system 
    * prompt injection and jailbreak: malicious instructions are injected into user prompts. It subverts models' safety feature 
### Chapter 6, RAG and Agents
#### RAG
* in applications, instructions are common to all queries, but context is specific for each query
* two strategies to construct context: RAG and agents
* RAG (retrieval-augmented generation): retrieve data from external data source/internet to address queries
    * RAG is a technique to construct context specific to each query instead of using the same context for all queries
    * Parkinson's Law: work expands so as to fill the time available for its completion
    * RAG allows a model to use only the most relevant information for each query
    * architecture: see below
        * retriever is often trained separately from the model: indexing and querying      
    ```
    prompt -------> retriever <------------external memory
        |               |
        v               v
        ----------> generative model
    ```
* retrieval ranks documents based on their relevance to a given query
    * term based vs. embedding based (sparse vs. dense)
    * term based find keywords in the documents
        * can be ranked by term frequency (TF)
        * inverse document frequency (IDF): the importance of a term is inversely linked to its frequency
        * BM25 algorithm
    * tokenization: breaking down phrase into tokens might lose their meaning (hot dog), instead can use n-gram
    * embedding based: embedding model + retriever
        * embedding model: convert query into an embedding
        * fetch data whose embeddings are closest to the query
        * *vector database*: find vectors closest to the query
        * search algo in vector database is hard, can use k-nearest neighbors, but computational intense
        * more details about vector db: https://zilliz.com/learn/vector-index
        * evaluate retrievers: context and recall
        * performance and cost
            * performance compromise of embeddings: added latency by query embedding and vector search might be minimal compared to the total RAG latency
            * generate embeddings, vector storage are costly
    * term based is faster than embedding based
    * trade off between indexing and querying: detailed query makes the results more accurate but time consuming to build
    * hybrid search combine term based and embedding based model, search for keywords first and rank them with embeddings
* retrieval optimization:
    * chunking:
        * split into equal length, or recursively into smaller chunks
        * overlapping ensures information is not cut in the middle
        * trade of of chunk size: smaller chunks -> more diverse info, but more chunks to process and store; larger -> preserve context and integrity
    * reranking: narrow down documents returned by the retriever
        * in context reranking (for models), exact ranks are less important compare to traditional ranking
    * query rewrite: supplement query with necessary context information
    * contextual retrieval: augment chunks with relevant context info such as metadata
        * tags, keywords
        * relationship with the original, longer documents
* RAG beyond text
    * multimodal embedding models convert different formats to embeddings to compare them
    * tabular data: generate SQL to help find answers
#### Agent
* agent use tools to perform a set of actions in an environment
* agents typically require more powerful models compared to non-agent uses: an agent with 95% accuracy that performs 10 steps will have an accuracy of 60%
* tools can fall into 3 categories: knowledge augmentation, capability extension and to act upon their environment
    * read and write actions
    * write actions are risky
    * function calling
* Q: chat GPT is a general purpose agent while most applications have defined workflows?
* To avoid fruitless execution, planning should be decoupled from execution
    * three components: one to generate plan, one to validate plan, one to execute plan 
    * can generate several plans in parallel
    * taking into consideration the intent of the user can help with planning (https://platform.openai.com/docs/guides/prompt-engineering/strategy-split-complex-tasks-into-simpler-subtasks)
    * foundation models as planners
        * many researchers believe autoregressive models can't plan
    * plan generation
    * function calling
    * planning granularity
    * execution sequence: sequence, parallel, loops, etc
    * reflection and error correction
#### Memory
* mechanism that allows a model to retain information
* three components
    * internal memory: data it is trained on
    * short-term memory: past messages in a conversations can be added as context, not persisted, limited by a model's context length
    * long-term memory: RAG, persisted across tasks
* memory system: memory management, memory retrieval
    * memory management: 
        * move short-term memory overflow to long-term memory
    * FIFO, first in first out, but beginning of a conversation might be important
    * remove redundancy in conversations thru AI-as-judge or summarization
## [Flow Engineering](https://www.youtube.com/watch?v=YpoK2L1EeJc)
* instead of calling LLMs once, applications usually need to call them multiple times
* design environment, eg. at Ford, the craftsmanship is in the workbench itself
* unlike knowledge work, decomposing knowledge work requires context sharing (pass information between people), which can be expensive and time-consuming
* limitations
    * LLMs are stateless; they don't remember chat history (reasoning models are improving on this aspect)
    * LLMs don't complain about errors -> engineers need to make judgement about quality of output
    * LLMs don't make good judgement off the shelf
* Flow engineering is the designing and optimizing cognitive factories
    * decompose tasks
    * design ergonomic prompts
    * design communication between LLMs
    * embed judgements into workflow
    * oversight and iteration
    * empower LLMs with tools and resource
    * tweak specifications of final output
* What tasks are not suited for flow engineering
    * require high degrees of flexibility (end to end RL models are better)
    * require a deep synthesis of information, eg. scientific research token workf
# Math topic
## Low rank transformation
# Curious topics
## A Mathematical Framework for Transformer Circuits (reverse engineer attention models)
## Will we run out of data? Limits of LLM scaling based on human-generated data
## Chinchilla scaling law: Training Compute-Optimal Large Language Models
# Bookmarks
* OpenAI tokenizer: https://platform.openai.com/tokenizer
* Will we run out of data? Limits of LLM scaling based on human-generated data
* AI Search Has A Citation Problem
* 2025 AI Engineering reading list: https://www.latent.space/p/2025-papers#%C2%A7section-voice
* Lilian Weng's blog posts on ML topics
* Shah and Bender-Envisioning Information Access Systems
# Project ideas
## agent that generate similarity queries that help with search
## Bare bone GPT with RAG on embedded systems
## Bookmark organizer
## How GPTZero works?
* https://scholar.google.ca/scholar?cites=6956709800612024780&as_sdt=2005&sciodt=0,5&hl=en
## Common Crawl
# Talks
## [Understand Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms and https://news.ycombinator.com/item?id=42966720)
* good at complex tasks such as solving puzzles, math, and coding tasks
* not necessary for summarization, translation, or knowledge-based question answering
* mechanisms
    * inference-time scaling: use more resources for inference to get better answers, such as prompt engineering techniques
    * RL: for DeepSeek Zero, reinforcement learning with human feedback, where models are rewarded by accuracy (leetcode compiler) and format
    * Supervised fine-tuning and RL: perform SFT before RL
## [Understanding and Using Supervised Fine-Tuning (SFT) for Language Models](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised)
## Agent Engineering in 2025 (+ Q&A on Reliable Agents)
## Building effective agents: https://www.anthropic.com/engineering/building-effective-agents 
## Q:
* what is the typical size of an embedding unit (text that are converted to embeddings)?