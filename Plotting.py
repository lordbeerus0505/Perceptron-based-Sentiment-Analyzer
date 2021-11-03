import matplotlib.pyplot as plt

""" 
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
"""


import numpy as np
# for each value of max iterations, creating list of answers
learning_rate_accuracies_perceptron = [
    [0.7272058823529411, 0.7308823529411764, 0.7279411764705882, 0.7382352941176471, 0.738235294117647, 0.7345588235294117],
    [0.7330882352941176, 0.7345588235294118, 0.7323529411764705, 0.7411764705882353, 0.7367647058823529, 0.7441176470588234],
    [0.7330882352941176, 0.7345588235294118, 0.7323529411764705, 0.7411764705882353, 0.7367647058823529, 0.7441176470588234],
    [0.7330882352941176, 0.7345588235294118, 0.7323529411764705, 0.7411764705882353, 0.7367647058823529, 0.7441176470588234],
    [0.7330882352941176, 0.7345588235294118, 0.7323529411764705, 0.7411764705882353, 0.7367647058823529, 0.7441176470588234],
    [0.7330882352941176, 0.7345588235294118, 0.7323529411764705, 0.7411764705882353, 0.7367647058823529, 0.7441176470588234],
]

learning_rate_accuracies_logistic = [
    [0.7573529411764706, 0.7661764705882352, 0.7705882352941178, 0.7794117647058825, 0.7823529411764707, 0.7882352941176471],
    [0.7669117647058823, 0.7720588235294118, 0.7860294117647059, 0.7816176470588235, 0.786764705882353, 0.7867647058823529],
    [0.7691176470588236, 0.7816176470588235, 0.7830882352941176, 0.7889705882352941, 0.7933823529411763, 0.7801470588235293],
    [0.7742647058823529, 0.7816176470588235, 0.7941176470588235, 0.7867647058823529, 0.7830882352941176, 0.775735294117647],
    [0.7860294117647059, 0.7919117647058823, 0.7882352941176471, 0.7808823529411766, 0.7764705882352941, 0.773529411764706],
    [0.7867647058823529, 0.7875, 0.7757352941176471, 0.775, 0.7735294117647059, 0.7654411764705882]
]


learning_rates = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2]
max_iter = [50, 100, 150, 300, 500, 1000]

for i in range(6):
    plt.plot(learning_rates, learning_rate_accuracies_perceptron[i])
    plt.scatter(learning_rates, learning_rate_accuracies_perceptron[i])
    plt.title("Perceptron: Accuracy vs Learning Rate for %s iterations"%max_iter[i])
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    # plt.show()
    plt.savefig('perceptron_'+str(i)+'.jpg')
    plt.clf()
for i in range(6):
    plt.plot(learning_rates, learning_rate_accuracies_logistic[i])
    plt.scatter(learning_rates, learning_rate_accuracies_logistic[i])
    plt.title("Logistic Regression: Accuracy vs Learning Rate for %s iterations"%max_iter[i])
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    # plt.show()
    plt.savefig('logistic_'+str(i)+'.jpg')
    plt.clf()