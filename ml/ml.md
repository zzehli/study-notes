# Models
## Series
### [Intro to Large Language Models, Andrej Karpathy](https://www.youtube.com/watch?v=zjkBMFhNj_g)
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
### Neural Network, 3Blue1Brown
#### [Chapter 1. neural network (multilayer perceptron)](https://youtu.be/aircAruvnKk?si=BBdfFO5u6oVUcr3c)
* neural network as a metaphor (example, recognize an image of digits):
* neuron: a thing that holds a number (the first layer in the neural network is the all pixels of the image, the last is the probability of the possible results); the values (normalized between 0 to 1) of neurons are called *activation*; neurons are also functions that take output from the previous layer and output an activation
* activation in one layer determines the activation in another layer, which is similar to how neuron works
* the middle layers are called hidden layer
* the goal is to have a neuron recognize one aspect of the image, eg. a horizontal edge
* from one layer to another, calculates the weighted sum of the activation, which also include a bias and a normalization function (sigmoid)
* *learning* refers to the process of finding the right weights and biases
#### [Chapter 2. Gradient descent, NN training](https://youtu.be/IHZwWFHWa-w?si=h6RbFvXM-zv9QmnF)
* start the training with random weights and bias, calculate the *cost*; the cost is small when the prediction is correct and large when wrong
* the cost measures the performance of the parameters
* to tell the model to perform better, minimize the cost function by telling the model where to move (towards a local minimum) based on the direction of the fastest descent, aka the *gradient* of the location
* the algorithm for computing the gradient efficiently is called *backpropagation*
#### [Chapter 3. Backpropagation](https://youtu.be/Ilg3gGewQ5U?si=Y1ge6ogfs8sFxBra)
* adjust weights, bias so that desired activation of neurons are achieved
* use gradient descent to adjust weights and bias; use backpropagation to calculate gradient
* each neuron is connected to every neuron in the next layer, each of these neurons require different adjustments to this neuron, summing them get us the desired change, propagate this process backwards becomes the intuition of backpropagation
* because backpropagation on each neuron is costly, we batch training set and perform the backpropagation on a batch; this is faster; this is called *stochastic gradient descent*
#### [Chapter 4. Backpropagation calculus](https://youtu.be/tIeHLnjs5U8?si=828RZzkAjBPHsiMO)
* use chain rule to backpropagate (cost as a function of previous layers' weights * activate - bias) how each weights and bias affect the cost function to achieve gradient descent
#### [Chapter 5. How LLM work](https://youtu.be/wjZofJX0v4M?si=-tTdjMPGNNMZ5ckH)
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
#### [Chapter 6. Attention in transformers](https://youtu.be/eMlx5fFNoYc?si=k8-PyFulowhBQ1-T)
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
### Neural Networks: Zero to Hero
#### [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0?si=pJ88FgwuDpLq2OgR)
* Backpropagation is the recursive application of chain rule backward thru the computational graph
* a mathematical model of neuron, artificial neuron: $$f(\sum_i w_ix_i + b)$$ where $w_ix_i$ is weighted input and $b$ is bias, and $f$ is an activation function (sigmoid, ReLU) that normalizes the weighted output
* see notes in [micrograd](micrograd.ipynb)
### Build a Large Language Model (From Scratch) (Sebastian Raschka)
#### Chapter 6, Finetuning for classification
* classification and instruction fine-tuning
##### Discussion
* why is the dataset under-sampled to include the same amount of spam and non-spam: more representation of of non-spam
    * if the dataset has 99% of non-spams, the model automatically get 99% accuracy regardless how good it does on spams
* instruction dataset FLAN: Finetuned Language Models Are Zero-Shot Learners
* sequence packing
* roughly, parts of LLM do: 1/3: semantic representation; 2/3: knowledge acquisition; 3/3 undo something 
* weight decay: nudge weights towards 0 by a small amount
* bigger model learn better with small amount of data
### Hugging Face LLM Course
#### Transformer models
* typical llm tasks: zero-shot classification, text generation, mask filling, named entity recognition (NER), question answering, summarization, translation
* history of transformers: transformers (2017) -> GPT (June, 2018) -> BERT (Oct, 2018) -> GPT-2 (Feb, 2019) -> BART/T5 (Oct, 2019) -> GPT-3 (May 2020)
* categories
    * GPT: autoregressive transformers
    * BERT: auto-encoding transformers
    * BART/T-5 sequence-to-sequence transformers
* pre-trained go thru transfer learning, which is supervised finetuning
* architecture of transformers:
    * encoder: build features from input
    * decoder: use features to generate output
    * they can be used independently; encoder-only models for sentence classification and NER; decoder-only models for text generation;
* attention layer: this layer will tell the model to pay specific attention to certain words in the sentence you passed it (and more or less ignore the others) when dealing with the representation of each word
    * originally designed for translation
    * the encoder uses the whole sentence while the decoder works sequentially
* checkpoints are weights
* encoder models: bi-directional attention, auto-encoding models
    * BERT models
    * output features/vectors representations of each word in the sentence, the value of the word is contextualized
    * self-attention mechanism -> contextual understanding
* decoder models: autoregressive models, next-word prediction
    * used in text-generation (causal language model, natural language generation)
    * **GPTs are decoder-only**
    * uni-direction: only have access to text that comes before (the words after it are masked)
    * masked self-attention
    * decoder also outputs feature vector/tensor
* sequence-to-sequence models: use both parts of the transformer models
    * BART and T5
    * best for summarization, translation and question-answering
    * encoder produces feature vectors, which is used as input for the decoder
    * the feature vectors is only used to produce the first word in the sequence (the rest is autoregressive)
    * can use a BERT and GPT together to construct a new model
#### inference process
*  How inference work with the `pipeline()` function : tokenizer -> model -> post processing
    1. get the original tokenizer from the model's checkpoints
        * transformer models take tensors as inputs, which can be thought of as arrays
        * tokenizer return tensors 
            * first tokenizer turns texts into input IDs, then torch turns those into tensors
            * Q: do model take tensors or input IDs as input? input IDs are not tensor. A. Not the same, model takes tensors
            ```
            <!-- https://huggingface.co/learn/llm-course/chapter2/5 -->
            sequence = "I've been waiting for a HuggingFace course my whole life."

            tokens = tokenizer.tokenize(sequence)
            # ids and input ids are not the same
            ids = tokenizer.convert_tokens_to_ids(tokens)

            input_ids = torch.tensor([ids])

            output = model(input_ids)
            ```
    2. get the model from checkpoints
        * the model outputs *hidden states* or *features*, which are vectors represent the model's understanding of the input
        * these features are input into other parts of the model, known as heads, which accomplish the specified task (classification, sentence completion, etc)
    3. the output of the model are logits, which is then turned into probabilities with activation function (softmax)
    4. use tokenizer one more time to convert the result back to text, if needed
```

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```
* model can be reconstructed with two files, a config file a a *state dictionary*, which contains weights 
* tokenizer 
    * tokenization uses subwords, which keeps the vocabulary small, and little unknown words
    * each model is governed by a unique tokenization rules, so it is model specific
    * two steps
        * split into tokens
        * turn tokens into input IDs
* batching
    * models expect batched inputs: more than one, can send one sequence (sentence) with `[ids, ids]`
    * tensors are rectangular, so to handle two sequences with different length, use *padding* fill in the blanks
    * when going thru an attention layer, the padding tokens also gets analyzed, use *attention masks* to avoid that 
* truncate sequences to the max length of the model with `sequence = sequence[:max_sequence_length]`
#### Finetuning
* process dataset (handle padding, etc)
    * output of tokenizer: input_id, attention_mask
    * batching with a *collate* function
* specify model, dataset, hyperparameters, and tokenizers for training
* define eval process (accuracy, f1)
1. train
```
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```
2. eval: 
```
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```
#### Causal Language Model from scratch
* causal language models are basically chatGPT
* smaller context window is faster to train and require less memory
## Reasoning
### Wei et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2022-3)
* "chain-of-thought prompting": <input, chain of thought, output>
* a chain of thought is a series of intermediate reasoning steps that lead to the final answer
* toc decompose difficult problems into multiple steps
* few-shot prompting existed before (Brown et al., 2020), but no reasoning steps are provided 
* cot works better with models with more parameters
### Yao et al. ReAct (2023)
* acting and reasoning
> ReAct prompts LLMs to generate both verbal reasoning traces and actions pertaining to a task in an
> interleaved manner, which allows the model to perform dynamic reasoning to create, maintain, and
> adjust high-level plans for acting (reason to act), while also interact with the external environments
> (e.g. Wikipedia) to incorporate additional information into reasoning (act to reason).
* this ReAct behavior is achieved thru few-shot in-context prompting, where each in-context example is a human trajectory of *actions*, *thoughts*, and *environment observations* to solve a task instance (Appendix C)
* previous attempts:   
    * ACT: only action, no thoughts
    * CoT: only thoughts, no actions (leads to hallucination)
    * Inner monologue, (Huang et al. 2022): do both, but limited to thoughts about task decomposition, no observation (what is achieved from actions)
### Yao et al. Tree of Thought (2023)
* the probability of a sequence $x$ in an autoregressive model: 
$$p_\theta(x) = \Pi^n_{i=1}(x[i] | x[1..i]) $$
This read: the probability of a LM $\theta$ producing language sequence $x$ is given by the product of probability of each token $x[i]$.
* to turn an input $x$ into output $y$ with LM: $$y \sim p_\theta(y | prompt_{IO}(x)) $$ we simplified as $$y \sim p_\theta^{IO}(y | x) $$
* Chain of thought can be represented as $$y \sim p_\theta^{CoT}(y | x, z_{1...n})$$ where $z$ are a chain of intermediate thoughts
* tree of thoughts construct a problem as a tree of state, the LM can traverse any branch connected with its parent branch. If a branch doesn't work, it backtracks to the parent branch to explore alternatives
## Post-training
### Training language models to follow instructions with human feedback (openai, 2022)
* this paper populated RLHF (Training language models to follow instructions
with human feedback). It shows a fine tuned model 1B model, InstructGPT, exhibit better instruction-following ability than 175B GPT-3
* *alignment*: the ability for models to follow instructions
    * helpful: follow instruction
    * honest: truthfulness (measure hallucination)
    * harmless
* method
    1. collection demonstration data, and train a supervised policy
    2. collection comparison data (between model outputs), and train a reward model
    3. optimize a policy against the reward model with PPO
* Other resources:
    * openAI blog post: https://openai.com/index/learning-to-summarize-with-human-feedback/
    * Chip Huyen: https://huyenchip.com/2023/05/02/rlhf.html
    * Nathan Lambert: https://huggingface.co/blog/rlhf
    * weights and biases: https://wandb.ai/ayush-thakur/RLHF/reports/Understanding-Reinforcement-Learning-from-Human-Feedback-RLHF-Part-1--VmlldzoyODk5MTIx
    * Yannic Kilcher: https://www.youtube.com/watch?v=vLTmnaMpQCs
## PyTorch
### Components (quick start tutorial)
* tensors: `x_data = torch.tensor([1, 2])
    * shape of the tensor, the last two digit in `tensor.size()` represents the shape of the inner most matrices, more see [this post](https://wandb.ai/vincenttu/intro-to-tensors/reports/A-Gentle-Intro-To-Tensors-With-Examples--VmlldzozMTQ2MjE5)
* dataset (preload) & dataloader (wraps an iterable around a dset)
    * use `transform` to modify features and use `target_transform` to modify labels
* build neural net: `torch.nn`
    * initialize nn in `__init__`
    * operation on input data in `forward`
        * model layers: `Flatten`, `Linear`, `Relu`
        * put layers together into a workflow with `Sequential`
        * run `softmax` got get logits
* automatic differentiation: `torch.autograd`:
> In a forward pass, autograd does two things simultaneously:
> 
>     run the requested operation to compute a resulting tensor
>     maintain the operation's gradient function in the DAG.
> 
> The backward pass kicks off when .backward() is called on the DAG root. autograd then:
> 
>     computes the gradients from each .grad_fn,
>     accumulates them in the respective tensor's .grad attribute
>     using the chain rule, propagates all the way to the leaf tensors.

* An important thing to note is that the graph is recreated from scratch
* optimization: optimization step adjust param to reduce model errors
    * algorithms such as Stochastic Gradient Descent and ADAM
* optimization loop consists of train loop and test loop, each iteration is an *epoch* (putting it all together: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#optimizing-model-parameters)

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
## see Chain-of-Thought, ReAct papers under Reasoning
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
    * An embedding is a numerical representation that aims to capture the meaning of the original data
    * An embedding is a vector
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
    * sparse retrieval are term based since a vector's length is the length of the vocabulary and it is 1 at the position of itself and 0 everywhere else
    * dense retrieval uses embeddings, where most elements of a vector aren't 0 
    * term based find keywords in the documents
        * can be ranked by term frequency (TF)
        * inverse document frequency (IDF): the importance of a term is inversely linked to its frequency (eg. terms like the, for, are frequent but not informative)
        * BM25 algorithm
    * tokenization: breaking down phrase into tokens might lose their meaning (hot dog), instead can use n-gram
    * embedding based: embedding model + retriever
        * embedding model: convert query into an embedding
        * fetch data whose embeddings are closest to the query
        * *vector database*: find vectors closest to the query
        * search algo in vector database is hard, can use k-nearest neighbors, but computational intense
        * more details about vector db: https://zilliz.com/learn/vector-index
        * evaluate retrievers: precision and recall
            * precision: what percentage of the documents retrieved are relevant to the query
            * recall: out of all documents related to the query, what percentage is retrieved
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
### Chapter 7, Finetuning
#### RAG & Finetuning
* prompt engineering, RAG and agents are all prompt-based method, while finetuning changes the model itself
* finetuning is mostly used to train models to improve its instruction-following ability (eg. style and format)
* when to do RAG vs. finetuning
* finetuning is a instance of *transfer learning*, by training a base model on tasks with abundant data, you can then transfer that knowledge to a target task
    * another application of transfer learning is feature-based transfer used in CV
* finetuning is *sample efficient*, where fewer training data can achieve good results
* can finetune a model for autoregressive or masked tasks, the latter is called *infilling finetuning*, used in editing and code debugging
* supervised finetuning: trained with instruction-response pairs
* preference finetuning requires comparative data (instruction, winning response, losing response)
* long-context finetuning makes a model's context window/length longer
* reasons to finetune
    * finetune a small model with data generated by a larger model to make the small model behave like the larger model. This is called *distillation* 
    * bias mitigation
    * specific output format: JSON, YAML, etc
    * other domain-specific tasks
* reasons not to
    * finetuning might degrade models' performance
    * finetuning a model is expensive, and will be outdated once a better model comes out
* use prompt-based techniques before attempting finetuning
* to decide between finetuning and RAG, depends on whether the problem is information-based or behavior-based
    * information based problems: outputs are factually wrong or dated --> RAG
        * finetuning or retrieval: https://arxiv.org/pdf/2312.05934 
        * RAG also reduce hallucination
    * behavior based: format, irrelevant answers --> finetuning
    * workflow for diagnosis and improving model performance:
        1. establish eval
        2. try prompting and add more examples to prompts
        3. RAG with term-based search
        4. RAG with embedding-based search
        5. finetuning models
        6. combine them
#### memory management
* A *trainable parameter* is a parameter that can be updated during finetuning; during finetuning, some or all of a model's parameters are updated
* training mechanism: backpropagation
    * forward and backward passes
    * during inference only forward pass; during training, both passes
* memory requirements
    * inference: $$N * M * 1.2$$
        * N is # of params
        * M is memory for each param
        * for 13B models, if each param is 2 bytes, then memory is 13 * 2 * 1.2 = 31.2GB
    * training:  Training memory = model weights + activations + gradients + optimizer states
        * (this example does not take into account activations) Imagine you’re updating all parameters in a 13B-parameter model using the Adam optimizer. Because each trainable parameter has three values for its gradient and optimizer states, if it takes two bytes to store each value, the memory needed for gradients and optimizer states will be: 13 billion × 3 × 2 bytes = 78 GB
    * numerical values are represented with float numbers, the standard is FP64, which uses 64 bits to represent a float, but it's memory consuming, so the more common format for model trainings are FP16 and FP32
        * formats with more bits are higher precision
    * quantization: reduce model's precision
        * what to quantize: what you can quantize without hurting performance too much, often weights (as opposed to activations)
        * when: during or post-training; post-training quantization (PTQ) is the most common
#### Finetuning techniques
* memory-efficient finetuning techniques makes finetuning more accessible
* parameter-efficient finetuning, or 
* partial finetuning, only finetune layers closest to the outcome, leave the first layers intact; but it's parameter inefficient (requires many parameters to train, ~25%)
* PEFT (parammeter efficient finetuning) add two adapters (also model layers) to the training model and only adjust these parameters
    * add more parameters (add latency) but only train 3% of trainable parameters
    * two kinds of PEFT, adapter-based and soft prompt-based (adapter-based is the one defined above)
    * soft prompt-based: modify how the model processes the input by introducing special trainable tokens (vectors)
        * prefix-tuning, prompt tuning
* LORA is the most popular PEFT technique
    * LORA is adapter-based
    * LORA incorporates additional parameters in a way that doesn’t incur extra inference latency
    * LORA achieve the same or better performance with full finetuning with only 0.0027% of trainable parameters for GPT3
    * while LORA works well for finetuning, pretraining with LORA only effective when model is small
    * LORA is applied on model's weight matrices: mostly commonly, query, key, value and output projection matrices
    * quantization reduce LORA memory usage even further
* model-merging is another set of finetuning techniques that are novel
    * how to combine them: adding, layer stacking and concatenation
    * similar to ensemble learning: If model merging typically involves mixing parameters of constituent models together, ensembling typically combines only model outputs while keeping each constituent model intact
* finetuning hyperparameters: 
    * learning rate: determines how fast the model’s parameters should change with each learning step
    * batch size: The batch size determines how many examples a model learns from in each step to update its weights (the more the better, limited by hardware)
    * epochs: An epoch is a pass over the training data. The number of epochs determines how many times each training example is trained on
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
## structured output (openai doc)
* for JSONs
* can be done via response_format and function calling
* use response_format for end user, function calling for tool use (general sql to be run by databases)
## Agents
### orchestrating agents (openai swarm): https://cookbook.openai.com/examples/orchestrating_agents
* llm models can call tools based on natural language. This is the basis of agents
* an agent can call multiple tools, the developer can use system prompt to instruct the agent in terms of which tool to use
* as the number of tools and judgement steps increase, it becomes hard for ai to judge, which is why tasks can be divided as multiple routines (agents) to be handled separately
* a basic agent definition:
```
class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []
```
* notice agent does not have its own message array. Messages are shared among agents
* handoff functions are function that call another agent
### Openai agents sdk (production version of swarm)
* The underlying mechanism is similar to swarm, where an *agent* is defined by instruction, model, and tools (including handoffs)
* handoffs become a separate param, but under the hood, it is still given to the agent as a tool
* agents don't have to use tools
* agent loop: the llm runs in a loop until `final_output` is present (else it run tool calls or handoffs)
    * final_output is when the output does not have tool calls or handoffs
* use guardrails to moderate content; exceptions are raised that will halt agent runs
## RAG
* original documents need to be stored, it can be stored separately or together with the embeddings: https://www.reddit.com/r/LangChain/comments/1eibcqw/document_storage_in_rag_solutions_separate_or/
* most vector db can store text chunks, see the comparison table: https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/
* source document retention from langchain: https://python.langchain.com/docs/concepts/retrievers/#source-document-retention
* it's not possible to convert embeddings to its original text because embeddings are compressed representation of data
* Q: what's the difference between embeddings in decoder only models and here? are both one-direction only?
### [GraphRAG paper](https://arxiv.org/abs/2404.16130)
* Question GraphRAG targets: RAG fails in query-focused summarization (QFS) tasks, tasks that aims to summarize the whole corpus instead of retrieve specific pieces of information
    * example: ”What are the key trends in how scientific discoveries are influenced by interdisciplinary research over the past decade?”
* GraphRAG
    * construct a knowledge graph with LLMs extracting entities and relationship for each information chunk
    * Use LLM to generate community-level summaries (summaries are generated at different levels as well, leaf-level, high-level etc)
    * when answer the query, use map reduce where each community generates a partial answers and then combined to generate a global answer
* comment: strong paper, well-written
## eval
### RAG evals
* RAG triad introduced by Snowflake's TruEra: https://truera.com/ai-quality-education/generative-ai-rags/what-is-the-rag-triad/
    * Answer relevance: is the answer relevant to the query?
    * Context relevance: is the retrieved context relevant to the query?
    * Groundedness: is the response supported by the context
* a helpful resource building a rag eval pipeline from ground up: https://huggingface.co/learn/cookbook/en/rag_evaluation
## langchain
### Architecture
* based on graph architecture of Google's Pregel
* state
    * state has a schema, but intermediate schemas are also permitted
    * update state through reducers, which is defined in schema, field by field
* nodes 
    * perform state transformation by taking the state as parameter and return state updates (field by field, by default new field values will override the old, unless reducers specify otherwise)
    * cache expensive nodes
    * `START` and `END` nodes
    * use `Command` when performing state update and conditional edge at the same time (useful for agent multi-agent handoffs when you need to decide which agent to hand to as well as pass information)
* edge
    * use conditional edges as a control flow for edges.
    * implement a conditional edge with a routing function that returns the outcome nodes and a `add_condition_edge` function, eg: `graph.add_conditional_edges(START, routing_function)`
    * conditional entry point
    * use `SEND` when graph structure isn't clear at run time, such as map reduce patterns
## Code Execution:
https://www.valentinog.com/blog/caging-the-agent/
https://simonwillison.net/tags/code-interpreter/
https://amirmalik.net/2025/03/07/code-sandboxes-for-llm-ai-agents
# Math topic
## Low rank transformation
# Curious topics
## A Mathematical Framework for Transformer Circuits (reverse engineer attention models)
## Will we run out of data? Limits of LLM scaling based on human-generated data
## Chinchilla scaling law: Training Compute-Optimal Large Language Models
## Memory requirement for LLM inference:
* `M = (P x (Q/8)) x 1.2` where P is parameters and Q is number of bits
* https://modal.com/blog/how-much-vram-need-inference and https://ksingh7.medium.com/calculate-how-much-gpu-memory-you-need-to-serve-any-llm-67301a844f21
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
## Visual Autoregressive (VAR) Modeling NeurIPS 2024 paper club
* VAE is replaced by VQ-VAE
* quantization (rounding in multi-dimension) is needed for transformer models
* predicting multiple tokens at once
```
my subjective recap of 4o image gen innovations:

super omnimodality (voice image text) encoder, enabling
autoregressive token generation with domain-specific diffusion decoder 
(no reference yet) much better text generation in images
in context learning/conversational image gen
training on ghibli (japan allows this)/web screenshots, including top memes
```
## Tracing the thoughts of LLMs paper club
* https://www.anthropic.com/research/tracing-thoughts-language-model
* background: https://buttondown.com/ainews/archive/ainews-anthropic-cracks-the-llm-genome-project/
* construct a replacement model (cross-layer transcoders, CLT) to reproduce the behavior of an LLM on a specific dataset and interpret that replacement model
* Sparse auto-encoder (SAE) for interpretation studies