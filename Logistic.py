"""
You may need to import necessary modules like numpy and pandas. However, you can't use any external
libraries such as sci-kit learn, etc. to implement logistic regression and the training of the logistic function.
The implementation must be done completely by yourself.

We are allowing you to use two packages from nltk for text processing: nltk.stem and nltk.tokenize. You cannot import
nltk in general, but we are allowing the use of these two packages only. We will check the code in your programs to
make sure this is the case and if other packages in nltk are used then we will deduct points from your assignment.
"""

"""
This is a Python class meant to represent the logistic model and any sort of feature processing that you may do. You 
have a lot of flexibility on how you want to implement the training of the logistic function but below I have listed 
functionality that should not change:
    - Arguments to the __init__ function 
    - Arguments and return statement of the train function
    - Arguments and return statement of the predict function 


When you want the program (logistic) to train on a dataset, the train function will only take one input which is the 
raw copy of the data file as a pandas dataframe. Below, is example code of how this is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Logistic()
    model.train(data) # Train the model on data.csv


It is assumed when this program is evaluated, the predict function takes one input which is the raw copy of the
data file as a pandas dataframe and produces as output the list of predicted labels. Below is example code of how this 
is done:

    data = pd.read_csv('data.csv', index_col=0)
    model = Logistic()
    predicted_labels = model.predict(data) # Produce predictions using model on data.csv

I have added several optional helper methods for you to use in building the pipeline of training the logistic function. 
It is up to your discretion on if you want to use them or add your own methods.
"""

import numpy as np
from nltk import stem
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
import pandas as pd
count=0
class Logistic():
    def __init__(self):
        """
        The __init__ function initializes the instance attributes for the class. There should be no inputs to this
        function at all. However, you can setup whatever instance attributes you would like to initialize for this
        class. Below, I have just placed as an example the weights and bias of the perceptron as instance attributes.
        """
        self.weights = None
        self.bias = None
        self.learning_rate = 0.01
        self.num_epochs = 1000
        self.word_freq = None
        self.sample_size = 0

    def vectorize(self, sentences, unique_words):
        """ 
            Creates a feature vector for every input using frequency of words.
        """
        outputVector=[]

        if self.word_freq == None:
            self.word_freq = {}
            for sentence in sentences:
                # import pdb; pdb.set_trace()
                # tokens = word_tokenize(sentence)
                for token in sentence:
                    if token not in self.word_freq.keys():
                        self.word_freq[token] = 1
                    else:
                        self.word_freq[token] += 1
            # import pdb; pdb.set_trace()
            # sort by frequency, we only want the top 1000 features
            # but the topmost features are pretty much just stop words, 
            # we only want ones that add to context hence skimming them out.
            self.word_freq = dict(sorted(self.word_freq.items(), key=lambda item: item[1], reverse=True))
            # the middle ground
            self.word_freq = {key: value for key, value in self.word_freq.items() if value <1800 and value >110}
            
            # dict_items = self.word_freq.items()
            # self.word_freq = list(dict_items)[500:1500]
            # import pdb; pdb.set_trace()

        for sentence in sentences:
            feature_vector = []
            for word in self.word_freq:
                # import pdb; pdb.set_trace()
                feature_vector.append(sentence.count(word))
            outputVector.append(feature_vector)
        # print(len(outputVector))
        return outputVector
         
    def unique(self, sentences):
        import re
        # Using a set as it makes sure the words remain unique
        words = set()
        stopCharacters=[",",":","-","_",";",".","?","[","]", "(", ")"]
        outputList = []
        cleaned_sentences = []
        tokenizer = RegexpTokenizer(r'\w+')
        for inputSentence in sentences:
            # import pdb; pdb.set_trace()
            inputSentence = tokenizer.tokenize(inputSentence)
            # inputSentence = re.split('; |, |\*|\n|\-| |\[|\]|\(|\)|\{|\}|\"|\'|\/|',inputSentence)
            for x in inputSentence:
                if not(x in words) and not(x in stopCharacters):
                    words.add(x)
                    outputList.append(x)
            cleaned_sentences.append(inputSentence)
        # import pdb; pdb.set_trace()
        return outputList, cleaned_sentences

    def feature_extraction(self, data):
        """
        Optional helper method to code the feature extraction function to transform the raw dataset into a processed
        dataset to be used in perceptron training.
        """
        
        sentences = data['Text'].to_list()
        unique_list, sentences = self.unique(sentences)
        features = self.vectorize(sentences, unique_list)
        return features

    def sgn_function(self, perceptron_input):
        """
        Optional helper method to code the sign function for the perceptron.
        """
        if perceptron_input>0.5:
            return 1
        return 0

    def logistic_loss(self, predicted_label, true_label, data_point):
        """
        Optional helper method to code the logistic function.
        """

        """ global count
        if predicted_label != true_label:
            count+=1
            if (count%100 == 0):
                print("updating weights, count =", count)
            self.weights += self.learning_rate*true_label*data_point
        return """

        # y = np.array(true_label)
        # h_x = np.array(predicted_label)
        # h_x = np.where(h_x==0, 0.00001, 0.99999)
        # log_h_x = np.log2(h_x)
        # log_1_h_x = np.log2(1-h_x)
        # loss = -y*log_h_x - (1-y)*log_1_h_x
        # cost = 1. / self.sample_size * loss
        
        errors = (true_label - predicted_label)

        prod = np.dot(errors, data_point) + self.weights*0.01/self.sample_size
        delta_w = (self.learning_rate * prod)
        return delta_w


    def stochastic_gradient_descent(self, data, label):
        """
        Optional helper method to compute a gradient update for a single point.
        """

    def sigmoid(self, z):  
        # print(z) 
        if z>100:
            return 1
        if z<-100:
            return 0
        return (1 / (1 + np.exp(-z)) )[0]

    def update_weights(self, new_weights):
        """
        Optional helper method to update the weights of the perceptron during stochastic gradient descent.
        """
        self.weights = new_weights

    def update_bias(self, new_bias):
        """
        Optional helper method to update the bias of the perceptron during stochastic gradient descent.
        """
        self.bias = new_bias

    def predict_labels(self, data_point):
        """
        Optional helper method to produce predictions for a single data point
        """
        return

    def train(self, labeled_data, learning_rate=None, max_epochs=None):
        """
        You must implement this function and it must take in as input data in the form of a pandas dataframe. This
        dataframe must have the label of the data points stored in a column called 'Label'. For example, the column
        labeled_data['Label'] must return the labels of every data point in the dataset. Additionally, this function
        should not return anything.

        The hyperparameters for training will be the learning rate and maximum number of epochs. Once you find the
        optimal values, update the default values for both the learning rate and max epochs keyword argument.

        The goal of this function is to train the logistic function on the labeled data. Feel free to code this
        however you want.
        """
        
        features = np.array(self.feature_extraction(labeled_data), dtype='float64')
        labels = np.array(labeled_data['Label'].to_list(), dtype='float64')
        # labels = np.where(labels<1,-1,labels)
        # Need to add the bias term as well! one more term
        bias = [[1] for x in range(len(features))]
        features = np.append(features, bias, axis=1)
        # self.weights = [0.0 for x in range(len(features[0]+1))]
        self.weights = np.random.rand(1,len(features[0]+1))
        # self.weights = np.array(self.weights)
        self.sample_size = len(features)

        """ 
            NEED TO MAKE SUCH CHANGES
            self.num_epochs = max_epochs
            self.learning_rate = learning_rate
        """




         # import pdb; pdb.set_trace()
        """ 
        #######################################################
        features = np.array([[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1],
        [0,1,1,0], [0,1,1,1], [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1], [1,1,0,0],
        [1,1,0,1], [1,1,1,0], [1,1,1,1]],dtype='float64')
        bias = [[1] for x in range(len(features))]
        # import pdb; pdb.set_trace()
        features = np.append(features, bias, axis=1)
        # if b,d true
        labels = [0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1]
        labels = pd.Series( (v for v in labels) ) 
        self.weights = np.random.rand(1,len(features[0]+1))[0]
        # self.weights = np.array(self.weights)
        ####################################################### 
        """
        for _ in range(self.num_epochs):
            results = []
            for i in range(len(features)):
                # self.stochastic_gradient_descent(features[i], labels[i])
                delta_w = np.zeros(len(features[i]))
                predicted_label = self.sgn_function(self.sigmoid(np.dot(self.weights,features[i])))
                delta_w = self.logistic_loss(predicted_label, labels[i], features[i])
                # import pdb; pdb.set_trace()
                results.append(delta_w)
                self.weights += delta_w
            # Using L2 norm to determine if above or below a threshold
            if norm(results) < 10**-6:
                # import pdb; pdb.set_trace()
                print("Stopped early at epoch: ", _)
                print("Norm is:", norm(results))
                break

        return

    def predict(self, data):
        predicted_labels = []
        """
        This function is designed to produce labels on some data input. It is required that it take in 3 inputs. The 
        first input is the data in the form of a pandas dataframe. The second and third inputs are the learning rate 
        and max iterations as two keyword arguments respectfully. Once you find the optimal values of the 
        hyperparameters, update the default values for each keyword argument to reflect those values.
        
        Finally, you must return the variable predicted_labels which should contain a list of all the 
        predicted labels on the input dataset. This list should only contain integers  that are either 0 (negative) or 1
        (positive) for each data point.
        
        The rest of the implementation can be fully customized.
        """
        # import pdb; pdb.set_trace()
        features = np.array(self.feature_extraction(data), dtype='float64')
        bias = [[1] for x in range(len(features))]
        features = np.append(features, bias, axis=1)
        
        for i in range(len(features)):
            prediction = self.sgn_function(self.sigmoid(np.dot(self.weights,features[i])))
            predicted_labels.append(prediction)
        # for i in range(len(features)):
        #     predicted_labels.append(self.sgn_function(features[i]))
        # import pdb; pdb.set_trace()
        return predicted_labels
