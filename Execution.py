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
    logistic and perceptron
    [
    [0.7955882352941176, 0.7963235294117648, 0.7948529411764705, 0.7897058823529413, 0.7801470588235294, 0.7786764705882353], 
    [0.7816176470588235, 0.7926470588235294, 0.7985294117647059, 0.8, 0.7926470588235295, 0.7889705882352941],
    [0.7963235294117648, 0.7904411764705882, 0.8007352941176471, 0.7794117647058824, 0.7772058823529412, 0.775],
    [0.7963235294117648, 0.7852941176470588, 0.8014705882352942, 0.7764705882352941, 0.775, 0.7764705882352941],
    [0.8, 0.7941176470588235, 0.7779411764705882, 0.775, 0.775, 0.775],
    [0.7977941176470588, 0.7911764705882353, 0.774264705882353, 0.775, 0.7735294117647058, 0.7742647058823529]
    ], [
    [0.7102941176470589, 0.7397058823529411, 0.7397058823529411, 0.7397058823529412, 0.7066176470588236, 0.694852941176470],
    [0.7036764705882353, 0.7389705882352942, 0.7419117647058824, 0.7375, 0.7095588235294118, 0.6970588235294117],
    [0.7029411764705882, 0.7382352941176471, 0.7426470588235294, 0.7426470588235294, 0.7102941176470589, 0.6963235294117648],
    [0.7029411764705882, 0.7382352941176471, 0.7426470588235294, 0.7426470588235294, 0.7102941176470589,0.7103526,],
    [0.7007352941176471, 0.7419117647058824, 0.7441176470588237, 0.7441176470588237, 0.7102941176470589, 0.6955882352941177]
    ]
    
    # Need to perform 5 Fold on this.
    
    trainData5Fold = []
    size = len(train_data)//5
    for i in range(5):
        trainData5Fold.append(train_data.iloc[size*i:size*(i+1)])
    
    p = Perceptron()
    lr = Logistic()

    learning_rate_perceptron = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    max_iter_perceptron = [300, 500, 800, 1000, 1360]
    perceptron_accuracies = []
    for j in range(6):
        for k in range(6):
            sigmaPerceptron = 0
            for i in range(5):   
                test_data = trainData5Fold[i]
                newTrainData = train_data.drop(test_data.index)
                p.train(newTrainData, learning_rate=learning_rate_perceptron[j], max_iter = max_iter_perceptron[k])
                predicted_lables = p.predict(test_data)
                acc = accuracyKFold(predicted_lables, test_data["Label"].to_list())
                sigmaPerceptron += acc

            avgPerceptron = sigmaPerceptron/5
            perceptron_accuracies.append(avgPerceptron)
            print('Average Accuracy for learning rate = %s, max_iter = %s is Perceptron: %s'%(learning_rate_perceptron[j], max_iter_perceptron[k], avgPerceptron))
    learning_rate_logistic = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25]
    max_epochs_logistic = [50, 100, 300, 500, 800, 1000]
    logistic_accuracies = []
    for j in range(6):
        for k in range(6):
            sigmaLogistic = 0
            for i in range(5):
                test_data = trainData5Fold[i]
                newTrainData = train_data.drop(test_data.index)
                lr.train(newTrainData, learning_rate=learning_rate_logistic[j], max_epochs= max_epochs_logistic[k])
                predicted_lables = lr.predict(test_data)
                acc = accuracyKFold(predicted_lables, test_data["Label"].to_list())
                sigmaLogistic += acc
            avgLogistic = sigmaLogistic/5
            logistic_accuracies.append(avgLogistic)
            print('Average Accuracy for learning rate = %s, max_iter = %s is Logistic: %s'%(learning_rate_logistic[j], max_epochs_logistic[k], avgLogistic))
    print(logistic_accuracies, perceptron_accuracies)
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

    # perceptron.train(train_data)
    logistic.train(train_data)

    # predicted_train_labels_perceptron = perceptron.predict(train_data)
    # predicted_test_labels_perceptron = perceptron.predict(test_data)
    # predicted_test_labels_unseen_perceptron = perceptron.predict(test_data_unseen)

    predicted_train_labels_logistic = logistic.predict(train_data)
    predicted_test_labels_logistic = logistic.predict(test_data)
    predicted_test_labels_unseen_logistic = logistic.predict(test_data_unseen)

    # print('\n\n-------------Perceptron Performance-------------\n')
    # # This command also runs the evaluation on the unseen test set
    # eval(train_data['Label'].tolist(), predicted_train_labels_perceptron, test_data['Label'].tolist(),
    #      predicted_test_labels_perceptron, test_data_unseen['Label'].tolist(), predicted_test_labels_unseen_perceptron)

    print('\n\n-------------Logistic Function Performance-------------\n')
    # This command also runs the evaluation on the unseen test
    eval(train_data['Label'].tolist(), predicted_train_labels_logistic, test_data['Label'].tolist(),
         predicted_test_labels_logistic, test_data_unseen['Label'].tolist(), predicted_test_labels_unseen_logistic)
