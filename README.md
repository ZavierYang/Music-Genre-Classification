# Importance
This is a project from a Machine learning course (subject). If you come to watch because of your course (subject) assignment, do not just copy and paste this code or just modify the variables name otherwise your score is possible to be penalised. Moreover, this project focuses on research ability. Therefore, in this repository, I just tried to implement multiple models to compare their performances. There is no guarantee that these model are the best. 

In this course, there is a in-class Kaggle competition about thus project. My final result is, 

    Public Leaderboard: 28 / 306 (9%)
    Private Leaderboard: 80 / 306 (26%)

**This leaderboard is calculated with approximately 30% of the test data. The final results will be based on the other 70%, so the final standings may be different.**

# Music-Genre-Classification Specification
The goal of this project is to build and critically analyse some supervised Machine Learning algorithms, to automatically identify the genre of a song on the basis of its audio, metadata and textual features. 

This project aims to reinforce the largely theoretical Machine Learning concepts around models, data, and evaluation covered in the lectures, by applying them to an open-ended problem. You will also have an opportunity  to practice your general problem-solving skills, written communication skills, and creativity.

## Dataset
Each song is represented through a large set of features, and listed in the features.csv files. Each song is labelled with a single genre tag, which is provided in the labels.csv files.
* train features.csv: Contains features of 7678 training instances.
* train labels.csv: Contains a single genre label for each training instance
* valid features.csv: Contains features of 450 validation instances.
* valid labels.csv: Contains a single genre label for each validation instance.
* test features.csv: Contains features of 428 test instances

## Feature
* Metadata features: For each song, we provide its title, loudness, tempo, key, mode, duration, and time_signature .
* Text features: For each song, we provide a list of tags representing the words that appeared in the lyrics of the song and are human annotated (such as ‘dance’, ‘love’, or ‘never).
* Audio features: We provide 148 pre-computed audio features that were pre-extracted from the 30 or 60 second snippets of each track, and capture timbre, chroma, and ‘Mel  Frequency Cepstral Coefficients’ (MFCC) aspects of the audio. Each feature is continuous and the values are not interpretable

## Label
There are eight labels in the dataset
1. Soul and Reggae
2. Pop 
3. Punk 
4. Jazz and Blues
5. Dance and Electronica
6. Folk
7. Classic Pop and Rock
8. Metal

## Task
Develop Machine Learning models which predict the music genre based on features. You will implement and analyze different Machine Learning models in their performance; and explore the utility of the different types of features for music genre prediction.

## Source
University of Melbourne COMP90049 Subject.

# Explanation of My Implementation
In this section I will explain the methods and models I used in this project.

## Pre-processing
The data contains two issues which will  cause unsatisfactory training. Firstly, feature  values in some metadata and audio are numerical, but their scopes are extremely different. For instance, the scope of vect_1 is 35.0 to 55.0 approximately, while vect_24 is 100  to 500 roughly. The scope difference will  cause training be slow or unable to converge effectively. Secondly, text features cannot be directly input into the model for training. Therefore, data need to be transform to make it into trainable data.
1. Normalization: It allows acceleration and rapid convergence during training.
2. Bag of Word (BOW): Pre-processing is required to enable models to train the text features since lyrics cannot be directly inputted to models for training. BOW can  convert each article or comment to be represented by a one-dimensional vector, and the length of each vector is the total number of words. By applying BOW, the music lyrics can be represented as a numeric vector. Subsequently, lyrics are able to be trained by model.

## Feature Selection Hypotheses
Different feature combinations are necessary because not all features are valid for model prediction. For example, trackId and tile in data are unique values so that they are not useful for prediction. Moreover, there are models that are more effective for learning specific features. Consequently, we need to figure out which classifier is the most accurate in diverse feature hypotheses and various models. In general, the public's perception of genres is more related to "music." Hence audio must be within the combination. In addition, lyrics may relate to genres, it should be used as training features as well. Whether metadata is effective for genre classification is unknown, but the contained data is more related to "music" characteristic such as loudness, tempo, and key. As a result, metadata will be trained with audio in some of the hypotheses. 

Hypotheses:

    1. Lyrics
    2. Metadata and audio
    3. Audio
    4. Lyrics, metadata, and audio
    5. Lyrics and audio

## Models
1. K-Nearest Neighbours (KNN): KNN is one of the traditional classifiers which the learning method is based on instances. A testing data point’s class will be determined by measuring the distance to testing data points.
2. Random Forest (RF): RF is a classifier that consists the methods of bagging and decision tree. Create several decision trees randomly which train different features set from training data. Finally, ensemble all trained trees together to predict  the test data class.
3. Multi-Layer Perceptron (MLP): MLP is a model composed of many perceptrons and connected to many layers. The  weight in the MLP can be updated through loss function and backpropagation to find which input features are significant so that all numerical features can be trained by MLP
4. Convolutional Neural Networks (CNN): The advantage of CNN is that it only extracts important features for MLP training rather than input all the data directly. Therefore, the training efficiency will be much better than MLP
5. Long Short-Term Memory (LSTM): LSTM can make use of data continuity. That is, we can use audio feature more effeciently. 

## Results
1. KNN: In KNN implementation, three hypotheses have been trained, and the best result is this task’s baseline result (Table 1) because it is the most basic within all implemented models. Besides, the data is not normalized, while  others model will normalize; the purpose is to compare the training difference with normalization data. The result indicates the distribution of data points for classification is not effective.

| hypothesis | Valid data accuracy | Test accuracy on Kaggle |
|-----------------|:-------------|:-------------|
| Lyrics | 30.45% | 26.57% |
| Metadata + audio | 34.45% | 33.60% (Baseline) |
| Lyrics + Metadata + audio | 34.45% | 33.60% (Baseline) |
