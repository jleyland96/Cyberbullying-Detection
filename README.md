# Cyberbullying-Detection
Final year project at Durham University - using Machine Learning and Deep Learning to detect Cyberbullying in Messages.

# Summary
## Context/Background 
Cyberbullying has affected an estimated 14.9% of high-school students in the last 12 months alone and is now as common as face-to-face bullying. This illustrates the potential and the need for a method of automatic cyberbullying detection using Machine Learning and Deep Learning.
## Aims
The aim of this project is to improve on the current state of the art for cyberbullying detection and evaluating if Deep Learning reliably out-performs traditional Machine Learning methods in this task. Additionally, the ability of a model to distinguish between different types of cyberbullying (e.g. racism, sexism) is critically assessed, along with the importance of a good quality dataset.
## Method 
Datasets were collected where used in related research and made publically available by the author; the accompanying results forming a benchmark in performance for this project. Other datasets were found online in public Git repositories. A number of Machine Learning models are implemented (SVM, Naive Bayes, Gradient Boosted Classifier etc.), then improved upon with Deep Learning methods (RNNs, CNNs, etc). The improvement in performance (if any) is assessed. Furthermore, cutting-edge methods such as ELMo word representations and new ideas such as custom loss functions are evaluated.
## Results 
High performance is achieved - above 0.76 F1 score on all four datasets in this project, including 0.8372 on one Twitter dataset. For the 2-class problem (detecting if cyberbullying is present or not), Deep Learning methods out-performed traditional Machine Learning methods by approximately 3%, however no improvement was seen on the 3-class problem (racism, sexism or neither). The running times and computational overhead associated with such complex Deep Learning models (such as using ELMo word representations) must be considered for such marginal performance improvements on this task.
## Conclusions 
The solution detects cyberbullying with rather high performance which renders the project a success, and it is concluded that Deep Learning methods distinguish between types of cyberbul- lying better than they can detect the presence of cyberbullying or not. However, a good underlying dataset is vital for success in this task, and the publicly available datasets are too small, likely due to the offensive nature of the messages which provides a deterrent for anyone wanting to gather such data. One might want to gather as much open-source data as possible, and combine these small datasets into one large corpus.


# Source Code
See below for information about the datasets in this project, and the python scripts I used to analyse them, clean them, and create Machine Learning and Deep Learning models on them.

## Datasets
cleaned_dixon.csv - cleaned dataset of 70,000 reddit comments with labels (1=Cyberbullying, 0=None). 
From https://github.com/EdwardDixon/Automation-and-Harassment-Detection

cleaned_twitter_1K.csv - cleaned dataset of 1,000 tweets (1=Cyberbullying, 0=None).
From https://github.com/chantelmariediaz/Predicting-Cyberbulling-on-Twitter

cleaned_tweets_16k.csv - cleaned dataset of 16,000 tweets (2=sexism, 1=racism, 0=neither racism nor sexism). 
From https://github.com/zeerakw/hatespeech

cleaned_tweets_16k_3class.csv - same dataset as above, except combined sexism and racism labels for binary task

## Python scripts
__dataset_analysis.py__ - extracts information from each dataset and cleans them

__ML_classifiers_combined.py__ - runs 10 Machine Learning classifiers on dataset of your choice (edit line 309), with numerous methods of feature extraction (edit line 306), such as GloVe, Avg. GloVe vector, Term Counts, TF, TF-IDF, character Bigrams and character Trigrams.

__DL_classifiers.py__ - main script for Deep Learning models. Support for the 2-class and 3-class data problems. Function provided with GloVe embeddings and learn-own-embeddings alternative. Also, functions provided for models that maxmimise F1 score directly. Also contains code used in the final year project demo, an interactive menu for loading saved models and training some supported models.

__DL_lstm_elmo.py__ - implementation for using ELMo contextualised word representations as features, and applying LSTM.

__DL_elmo.py__ - different implementation for using ELMo contextualised word representations as features, and applying a dense network.

__ML_ensemble.py__ - creating an ensemble of ML classifiers and majority voting on input instances.

__multichannel_cnn.py__ - implementation for multichannel convolutional network. 3 convolutions in parallel, outputs concatenated and classified. Both GloVe and One-hot support.


## Directories
__Design Report__ - Initial solution design paper from January 2019.

__Final Paper__ - Final research paper. Structure: Abstract, Introduction, Related Work, Solution, Results, Evaluation, Concluision. Find abstract above in README.

__Literature Review__ - Review of existing related work from September 2018, before the project began.

__Project Plan__ - Initial project plan from Summer 2018.

__Project Presentation__ - Presentation from February 2019, detailing an introduction to the project and my progress so far.

__Project Logbook__ - Compiled list of notes made in project meeting. Also contains a wealth of experimentation results from the duration of the project.

__Screenshots__ - A list of graphs for each dataset, for instance length distribution, and naughty word frequency distribution.

__saved_models__ - some of my best Deep Learning models saved, to be loaded and retrained in DL_classifiers.py and my project demo.


## Other files
__Corpus of naughty words__ - naughty_words.txt

__Dirty/raw/redundant datasets__ - cleaned_formspring.csv, cleaned_text_messages.csv, dixon_train_data.csv, formspring.csv, processed_tweets_16k.csv, processed_tweets_16k_3class.csv, processed_tweets_16k_copy.csv, tweets_7K_raw.csv, twitter_16K_raw.csv, twitter_16K.csv, twitter_1K.csv, twitter_7K.csv

__Early ML classifier python scripts__ - glove_classifier.py, naive_norm_classifier.py, ngram_classifier.py, term_count_classifier.py, term_freq_classifier.py

__Slurm files__ - my_DL_slurm, my_ML_slurm, etc....

__Redundant scripts__ - demo.py, get_tweets_example.py, get_tweets.py, 

