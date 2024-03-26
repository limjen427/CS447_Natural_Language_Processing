# CS447 Natural Language Processing

## Overview
Welcome to the repository for my completed Natural Language Processing (NLP) course assignments. This repository showcases my journey through understanding and implementing diverse NLP techniques and methodologies. Each assignment delves into different facets of NLP, ranging from foundational language modeling to advanced tasks like semantic role labeling and discourse analysis. Through practical implementation and exploration, this repository showcases my proficiency in harnessing NLP algorithms to extract insights from text data and tackle real-world language understanding challenges. Explore the assignments to witness the evolution of my skills in NLP, from basic concepts to cutting-edge neural models and ethical considerations in language technology.

## Course Information
- **Programming Language**: Python (primarily PyTorch and TensorFlow)
- **Class Goals**:
  - Develop a solid understanding of Natural Language Processing (NLP) fundamentals, including language modeling, morphological analysis, and semantic interpretation.
  - Gain hands-on experience implementing NLP algorithms for tasks such as part-of-speech tagging, named entity recognition, and sentiment analysis.
  - Explore advanced NLP techniques such as neural network architectures for sequence labeling tasks, machine translation, and text generation.
  - Understand the principles behind word embeddings and their applications in semantic similarity, document classification, and information retrieval.
  - Learn about modern NLP architectures, such as transformers, and their applications in tasks such as language modeling, text classification, and question answering.
  - Discuss ethical considerations in NLP, including bias and fairness, privacy concerns, and the responsible deployment of NLP technologies.

## Assignments

### Assignment 1: Language Models & Morphological Transduction
- 1) Build some n-gram language models on a corpus of Wikipedia articles.
- 2) Design a finite-state transducer for verb conjugation in Spanish.

### Assignment 2: Word Embeddings & Text Classification with Neural Networks
- 1) Train word embeddings using the continuous-bag-of-words (CBOW) method, a technique based on the word2vec paradigm, which involves predicting a word based on the embeddings of surrounding words within a given context to gain insights into dense representations of words in a continuous vector space.
- 2) Implement text classification using a Convolutional Neural Network (CNN). Specifically, Build a classifier to detect the sentiment of movie reviews utilizing the IMDb movie reviews dataset. Explore the application of deep learning techniques in natural language understanding tasks.

### Assignment 3: Neural Machine Translation
Perform machine translation using two deep learning approaches: a Recurrent Neural Network (RNN) and a Transformer. Train sequence-to-sequence models for Spanish-to-English translation. Resources for more details: [1](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) [2](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) [3](https://arxiv.org/pdf/1409.0473.pdf)
- 1) Implement a recurrent model for machine translation and training. Resources: [Attention paper](https://arxiv.org/pdf/1409.0473.pdf), [Explanation of LSTM's & GRU's](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21), [Attention explanation 1](https://towardsdatascience.com/attention-in-neural-networks-e66920838742), [Attention explanation 2](https://towardsdatascience.com/attention-and-its-different-forms-7fc3674d14dc)
- 2) Implement a transformer model for machine translation and training. Resources: [1](https://arxiv.org/pdf/1706.03762.pdf)

### Assignment 4:Dependency Parsing
Build a neural transition-based dependency parser, based on the paper [A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf)

