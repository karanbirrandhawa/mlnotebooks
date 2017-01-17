# Machine Learning, Data Mining and More

This repository is a collection of python notebooks that were helpful towards me learning about machine learning, statistical modelling, data mining, etc. They mostly consist of implementations of core parts of the algorithms and tools associated with each. 

## Techniques

I wanted to become very familiar with the techiques that data scientists and machine learning engineers used on a regular basis so I've listed them out to go through them one by one. The implementations for these techniques sometimes go into mathematical depth and often implement them from scratch. 

### Neural Networks

Neural networks 

**[The Perceptron](NeuralNetworks/notebooks/perceptron.ipynb)**

* An important technique that rests at the core of a neural network.
* The single layer perceptron algorithm allows us to build and train a linear classifier.
* This unfortunately means that we require data that is linearly seperable. If not, this technique sucks.

**[The Multi-Layer Perceptron (Feedforward Network)](NeuralNetworks/notebooks/multilayer_perceptron.ipynb)**

* More useful than single layer perceptron for non-linear data.
* Utilizes backpropogation algorithm.

**Convolutional Neural Network**

**Recurrent Neural Network**

* [Recurrent Neural Network Language Models](NeuralNetworks/notebooks/RNN_language_model.ipynb)
* Really useful for problems in which training data has recognizable sequences

**Restricted Boltzmann Machine**

### Bayesian Inference and Networks

This is a statistical approach that uses Bayes theorem to get predictions from data. 

**[Naive Bayes Classifier](Bayesian/notebooks/naive_bayesian_classifier.ipynb)**

- Probablistic classifier. Operates under the assumption that features are independent. 
- Several variations: gaussian, multinomial, binomial.

**Relevant Vector Machine**

**Bayesian Networks**

### Random Forest and Ensemble Learning

**Decision Tree Learning**

**Bootstrap Aggregating (Tree Bagging)**

**Random Forests (Random Subspace Method)**

### Support Vector Machines

These are supervised learning models useful for classification. They view each data point as a p-dimensional vector and are curious whether we can seperate these points into a (p-1)-dimensional hyperplane. 

**Linear SVM (Hard Margin)**

**Linear SVM (Soft Margin)**

**Non-linear SVM with Polynomial Kernel Trick**

**Non-linear SVM with Gaussian Radial Basis Kernel Trick**

**Support Vector Clustering (SVC)****

### Clustering

**Hierarchical Clustering**

**K-Means Clustering**

**K-Nearest-Neighbors**

### Dimensionality Reduction

**Primary Components Analysis (PCA)**

**Kernal PCA**

**Linear Discriminent Analysis (LDA)**

### Markov Chains

**Discrete-Time Markov Chains**

**Continuous-Time Markov Chains**

### Genetic Programming 

**Genetic Algorithms**

## Appendix

### Picking the Right ML Technique

Because of the sheer number of techniques that I invested in learning I had a number of options available when working on a project that required machine learning. I want to fill this section with thoughts on how to pick the best technique for a particular problem.

### Unsupervised Learning 

Unsupervised learning isn't covered to nearly as much depth as I would like here. Just making a note to self to get on that ASAP.

### TODO

- Finish all implementations 
- Add more content to the readme 
- Do some general purpose projects that put the techniques above into action
