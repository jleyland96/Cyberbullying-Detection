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
## Datasets
cleaned_dixon.csv - cleaned dataset of 70,000 reddit comments with labels (1=Cyberbullying, 0=None). 
From https://github.com/EdwardDixon/Automation-and-Harassment-Detection

cleaned_twitter_1K.csv - cleaned dataset of 1,000 tweets (1=Cyberbullying, 0=None).
From https://github.com/chantelmariediaz/Predicting-Cyberbulling-on-Twitter

cleaned_twitter_16k.csv - cleaned dataset of 16,000 tweets (2=sexism, 1=racism, 0=neither racism nor sexism
From https://github.com/zeerakw/hatespeech

cleaned_twitter_16k_3class.csv - same dataset as above, except combined sexism and racism labels for binary task
