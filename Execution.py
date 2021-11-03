import pandas as pd

"""
Execution.py is for evaluating your models on the datasets available to you. You can use 
this program to test the accuracy of your models by calling it in the following way:
    
    import Execution
    Execution.eval(o_train, p_train, o_test, p_test)
    
In the sample code, o_train is the observed training labels, p_train is the predicted training labels, o_test is the 
observed test labels, and p_test is the predicted test labels. 
"""

def split_dataset(all_data):
    train_data = None
    test_data = None
    """
    This function will take in as input the whole dataset and you will have to program how to split the dataset into
    training and test datasets. These are the following requirements:
        -The function must take only one parameter which is all_data as a pandas dataframe of the raw dataset.
        -It must return 2 outputs in the specified order: train and test datasets
        
    It is up to you how you want to do the splitting of the data.
    """
    # Using Pandas sampling here
    # splitting as 20% test 80% train
    # defining a random state as 32 since not mentioned in question, this will shuffle the data well
    all_data = all_data.sample(frac=1, random_state=18)
    test_data = all_data.sample(frac=0.20, random_state=32)
    train_data = all_data.drop(test_data.index)
    """ 
    NOTE: This section was evaluated directly on Google Colab and the results were used
    in deciding the hyper parameter values for both the models.
    The results were as follows :
    Average Accuracy for learning rate = 0.01, max_iter = 50 is Perceptron: 0.7272058823529411, Logistic: 0.7573529411764706
    Average Accuracy for learning rate = 0.01, max_iter = 100 is Perceptron: 0.7330882352941176, Logistic: 0.7669117647058823
    Average Accuracy for learning rate = 0.01, max_iter = 150 is Perceptron: 0.7330882352941176, Logistic: 0.7691176470588236
    Average Accuracy for learning rate = 0.01, max_iter = 300 is Perceptron: 0.7330882352941176, Logistic: 0.7742647058823529
    Average Accuracy for learning rate = 0.01, max_iter = 500 is Perceptron: 0.7330882352941176, Logistic: 0.7860294117647059
    Average Accuracy for learning rate = 0.01, max_iter = 1000 is Perceptron: 0.7330882352941176, Logistic: 0.7867647058823529
    Average Accuracy for learning rate = 0.025, max_iter = 50 is Perceptron: 0.7308823529411764, Logistic: 0.7661764705882352
    Average Accuracy for learning rate = 0.025, max_iter = 100 is Perceptron: 0.7345588235294118, Logistic: 0.7720588235294118
    Average Accuracy for learning rate = 0.025, max_iter = 150 is Perceptron: 0.7345588235294118, Logistic: 0.7816176470588235
    Average Accuracy for learning rate = 0.025, max_iter = 300 is Perceptron: 0.7345588235294118, Logistic: 0.7816176470588235
    Average Accuracy for learning rate = 0.025, max_iter = 500 is Perceptron: 0.7345588235294118, Logistic: 0.7919117647058823
    Average Accuracy for learning rate = 0.025, max_iter = 1000 is Perceptron: 0.7345588235294118, Logistic: 0.7875
    Average Accuracy for learning rate = 0.05, max_iter = 50 is Perceptron: 0.7279411764705882, Logistic: 0.7705882352941178
    Average Accuracy for learning rate = 0.05, max_iter = 100 is Perceptron: 0.7323529411764705, Logistic: 0.7860294117647059
    Average Accuracy for learning rate = 0.05, max_iter = 150 is Perceptron: 0.7323529411764705, Logistic: 0.7830882352941176
    Average Accuracy for learning rate = 0.05, max_iter = 300 is Perceptron: 0.7323529411764705, Logistic: 0.7941176470588235
    Average Accuracy for learning rate = 0.05, max_iter = 500 is Perceptron: 0.7323529411764705, Logistic: 0.7882352941176471
    Average Accuracy for learning rate = 0.05, max_iter = 1000 is Perceptron: 0.7323529411764705, Logistic: 0.7757352941176471
    Average Accuracy for learning rate = 0.075, max_iter = 50 is Perceptron: 0.7382352941176471, Logistic: 0.7794117647058825
    Average Accuracy for learning rate = 0.075, max_iter = 100 is Perceptron: 0.7411764705882353, Logistic: 0.7816176470588235
    Average Accuracy for learning rate = 0.075, max_iter = 150 is Perceptron: 0.7411764705882353, Logistic: 0.7889705882352941
    Average Accuracy for learning rate = 0.075, max_iter = 300 is Perceptron: 0.7411764705882353, Logistic: 0.7867647058823529
    Average Accuracy for learning rate = 0.075, max_iter = 500 is Perceptron: 0.7411764705882353, Logistic: 0.7808823529411766
    Average Accuracy for learning rate = 0.075, max_iter = 1000 is Perceptron: 0.7411764705882353, Logistic: 0.775
    Average Accuracy for learning rate = 0.1, max_iter = 50 is Perceptron: 0.738235294117647, Logistic: 0.7823529411764707
    Average Accuracy for learning rate = 0.1, max_iter = 100 is Perceptron: 0.7367647058823529, Logistic: 0.786764705882353
    Average Accuracy for learning rate = 0.1, max_iter = 150 is Perceptron: 0.7367647058823529, Logistic: 0.7933823529411763
    Average Accuracy for learning rate = 0.1, max_iter = 300 is Perceptron: 0.7367647058823529, Logistic: 0.7830882352941176
    Average Accuracy for learning rate = 0.1, max_iter = 500 is Perceptron: 0.7367647058823529, Logistic: 0.7764705882352941
    Average Accuracy for learning rate = 0.1, max_iter = 1000 is Perceptron: 0.7367647058823529, Logistic: 0.7735294117647059
    Average Accuracy for learning rate = 0.2, max_iter = 50 is Perceptron: 0.7345588235294117, Logistic: 0.7882352941176471
    Average Accuracy for learning rate = 0.2, max_iter = 100 is Perceptron: 0.7441176470588234, Logistic: 0.7867647058823529
    Average Accuracy for learning rate = 0.2, max_iter = 150 is Perceptron: 0.7441176470588234, Logistic: 0.7801470588235293
    Average Accuracy for learning rate = 0.2, max_iter = 300 is Perceptron: 0.7441176470588234, Logistic: 0.775735294117647
    Average Accuracy for learning rate = 0.2, max_iter = 500 is Perceptron: 0.7441176470588234, Logistic: 0.773529411764706
    Average Accuracy for learning rate = 0.2, max_iter = 1000 is Perceptron: 0.7441176470588234, Logistic: 0.7654411764705882
    # Need to perform 5 Fold on this.

    trainData5Fold = []
    size = len(train_data)//5
    for i in range(5):
        trainData5Fold.append(train_data.iloc[size*i:size*(i+1)])
    
    p = Perceptron()
    lr = Logistic()

    learning_rate = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2]
    max_iter = [50, 100, 150, 300, 500, 1000]

    for j in range(6):
        for k in range(6):
            sigmaPerceptron = 0
            sigmaLogistic = 0
            for i in range(5):   
                test_data = trainData5Fold[i]
                newTrainData = train_data.drop(test_data.index)
                p.train(newTrainData, learning_rate=learning_rate[j], max_iter = max_iter[k])
                predicted_lables = p.predict(test_data)
                acc = accuracyKFold(predicted_lables, test_data["Label"].to_list())
                sigmaPerceptron += acc

                lr.train(newTrainData, learning_rate=learning_rate[j], max_epochs= max_iter[k])
                predicted_lables = lr.predict(test_data)
                acc = accuracyKFold(predicted_lables, test_data["Label"].to_list())
                sigmaLogistic += acc
                # print(acc)
            avgPerceptron = sigmaPerceptron/5
            avgLogistic = sigmaLogistic/5
            print('Average Accuracy for learning rate = %s, max_iter = %s is Perceptron: %s, Logistic: %s'%(learning_rate[j], max_iter[k], avgPerceptron, avgLogistic))
    import pdb; pdb.set_trace() 
    """

    return train_data, test_data


def accuracyKFold(orig, pred):
    num = len(orig)
    if (num != len(pred)):
        print('Error!! Num of labels are not equal.')
        return
    match = 0
    for i in range(len(orig)):
        o_label = orig[i]
        p_label = pred[i]
        if (o_label == p_label):
            match += 1
    return (float(match) / num)

"""
This function should not be changed at all.
"""
def eval(o_train, p_train, o_val, p_val, o_test, p_test):
    print('\nTraining Accuracy Result!')
    accuracy(o_train, p_train)
    print('\nTesting Accuracy Result!')
    accuracy(o_val, p_val)
    print('\nUnseen Test Set Accuracy Result!')
    accuracy(o_test, p_test)

"""
This function should not be changed at all.
"""
def accuracy(orig, pred):
    num = len(orig)
    if (num != len(pred)):
        print('Error!! Num of labels are not equal.')
        return
    match = 0
    for i in range(len(orig)):
        o_label = orig[i]
        p_label = pred[i]
        if (o_label == p_label):
            match += 1
    print('***************\nAccuracy: '+str(float(match) / num)+'\n***************')


if __name__ == '__main__':
    """
    The code below these comments must not be altered in any way. This code is used to evaluate the predicted labels of
    your models against the ground-truth observations.
    """
    from Perceptron import Perceptron
    from Logistic import Logistic
    all_data = pd.read_csv('data.csv', index_col=0)
    train_data, test_data = split_dataset(all_data)

    # placeholder dataset -> when we run your code this will be an unseen test set your model will be evaluated on
    test_data_unseen = pd.read_csv('test_data.csv', index_col=0)

    perceptron = Perceptron()
    logistic = Logistic()

    perceptron.train(train_data)
    logistic.train(train_data)

    predicted_train_labels_perceptron = perceptron.predict(train_data)
    predicted_test_labels_perceptron = perceptron.predict(test_data)
    predicted_test_labels_unseen_perceptron = perceptron.predict(test_data_unseen)

    predicted_train_labels_logistic = logistic.predict(train_data)
    predicted_test_labels_logistic = logistic.predict(test_data)
    predicted_test_labels_unseen_logistic = logistic.predict(test_data_unseen)

    print('\n\n-------------Perceptron Performance-------------\n')
    # This command also runs the evaluation on the unseen test set
    eval(train_data['Label'].tolist(), predicted_train_labels_perceptron, test_data['Label'].tolist(),
         predicted_test_labels_perceptron, test_data_unseen['Label'].tolist(), predicted_test_labels_unseen_perceptron)

    print('\n\n-------------Logistic Function Performance-------------\n')
    # This command also runs the evaluation on the unseen test
    eval(train_data['Label'].tolist(), predicted_train_labels_logistic, test_data['Label'].tolist(),
         predicted_test_labels_logistic, test_data_unseen['Label'].tolist(), predicted_test_labels_unseen_logistic)
