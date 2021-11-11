import matplotlib.pyplot as plt

""" 
    Average Accuracy for learning rate = 0.01, max_iter = 50 is Perceptron: 0.7536764705882353, Logistic: 0.7595588235294117
Average Accuracy for learning rate = 0.01, max_iter = 100 is Perceptron: 0.7536764705882353, Logistic: 0.775
Average Accuracy for learning rate = 0.01, max_iter = 150 is Perceptron: 0.7536764705882353, Logistic: 0.7808823529411765
Average Accuracy for learning rate = 0.01, max_iter = 300 is Perceptron: 0.7536764705882353, Logistic: 0.786764705882353
Average Accuracy for learning rate = 0.01, max_iter = 500 is Perceptron: 0.7536764705882353, Logistic: 0.7919117647058823
Average Accuracy for learning rate = 0.01, max_iter = 1000 is Perceptron: 0.7536764705882353, Logistic: 0.7911764705882354
Average Accuracy for learning rate = 0.025, max_iter = 50 is Perceptron: 0.7551470588235294, Logistic: 0.7772058823529412
Average Accuracy for learning rate = 0.025, max_iter = 100 is Perceptron: 0.7536764705882353, Logistic: 0.7845588235294118
Average Accuracy for learning rate = 0.025, max_iter = 150 is Perceptron: 0.7536764705882353, Logistic: 0.7845588235294118
Average Accuracy for learning rate = 0.025, max_iter = 300 is Perceptron: 0.7536764705882353, Logistic: 0.7926470588235295
Average Accuracy for learning rate = 0.025, max_iter = 500 is Perceptron: 0.7536764705882353, Logistic: 0.7926470588235294
Average Accuracy for learning rate = 0.025, max_iter = 1000 is Perceptron: 0.7536764705882353, Logistic: 0.7823529411764707
Average Accuracy for learning rate = 0.05, max_iter = 50 is Perceptron: 0.7455882352941176, Logistic: 0.7830882352941176
Average Accuracy for learning rate = 0.05, max_iter = 100 is Perceptron: 0.7470588235294118, Logistic: 0.7897058823529411
Average Accuracy for learning rate = 0.05, max_iter = 150 is Perceptron: 0.7470588235294118, Logistic: 0.7926470588235295
Average Accuracy for learning rate = 0.05, max_iter = 300 is Perceptron: 0.7470588235294118, Logistic: 0.7897058823529413
Average Accuracy for learning rate = 0.05, max_iter = 500 is Perceptron: 0.7470588235294118, Logistic: 0.7830882352941176
Average Accuracy for learning rate = 0.05, max_iter = 1000 is Perceptron: 0.7470588235294118, Logistic: 0.775735294117647
Average Accuracy for learning rate = 0.075, max_iter = 50 is Perceptron: 0.7551470588235294, Logistic: 0.7823529411764706
Average Accuracy for learning rate = 0.075, max_iter = 100 is Perceptron: 0.7529411764705882, Logistic: 0.7926470588235295
Average Accuracy for learning rate = 0.075, max_iter = 150 is Perceptron: 0.7529411764705882, Logistic: 0.7919117647058823
Average Accuracy for learning rate = 0.075, max_iter = 300 is Perceptron: 0.7529411764705882, Logistic: 0.7823529411764707
Average Accuracy for learning rate = 0.075, max_iter = 500 is Perceptron: 0.7529411764705882, Logistic: 0.7764705882352942
Average Accuracy for learning rate = 0.075, max_iter = 1000 is Perceptron: 0.7529411764705882, Logistic: 0.7713235294117646
Average Accuracy for learning rate = 0.1, max_iter = 50 is Perceptron: 0.7551470588235294, Logistic: 0.7867647058823529
Average Accuracy for learning rate = 0.1, max_iter = 100 is Perceptron: 0.7566176470588235, Logistic: 0.7919117647058824
Average Accuracy for learning rate = 0.1, max_iter = 150 is Perceptron: 0.7566176470588235, Logistic: 0.7911764705882354
Average Accuracy for learning rate = 0.1, max_iter = 300 is Perceptron: 0.7566176470588235, Logistic: 0.7794117647058824
Average Accuracy for learning rate = 0.1, max_iter = 500 is Perceptron: 0.7566176470588235, Logistic: 0.775
Average Accuracy for learning rate = 0.1, max_iter = 1000 is Perceptron: 0.7566176470588235, Logistic: 0.7727941176470589
Average Accuracy for learning rate = 0.2, max_iter = 50 is Perceptron: 0.7529411764705882, Logistic: 0.7933823529411764
Average Accuracy for learning rate = 0.2, max_iter = 100 is Perceptron: 0.7529411764705882, Logistic: 0.7845588235294118
Average Accuracy for learning rate = 0.2, max_iter = 150 is Perceptron: 0.7529411764705882, Logistic: 0.7772058823529411
Average Accuracy for learning rate = 0.2, max_iter = 300 is Perceptron: 0.7529411764705882, Logistic: 0.7713235294117646
Average Accuracy for learning rate = 0.2, max_iter = 500 is Perceptron: 0.7529411764705882, Logistic: 0.7727941176470589
Average Accuracy for learning rate = 0.2, max_iter = 1000 is Perceptron: 0.7529411764705882, Logistic: 0.7727941176470589
"""


import numpy as np

logistic_accuracies, perceptron_accuracies = [
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

perceptron_train_accuracies, logistic_train_accuracies = [
    [0.7458823529411764, 0.7469176470325257, 0.7558292398125176, 0.7779411764705882, 0.7802352923517265, 0.8155882352941176],
    [0.7573529411764706, 0.7659292194635176, 0.7757352941176471, 0.7764705882352941, 0.7779411764705882, 0.8132352941176471],
    [0.7316176470588235, 0.7657352941176471, 0.7846318420917684, 0.8132352941176471, 0.8073529411764706, 0.8735294117647059],
    [0.7316176470588235, 0.7846318420917684, 0.8073529411764706, 0.8485123535239721, 0.8132352941176471, 0.8654411764705883],
    [0.7323529411764705, 0.7779411764705882, 0.8335294117647059, 0.8654411764705883, 0.8470588235294118, 0.8639705882352942],
], [
    [0.8514705882352941, 0.8816176470588235, 0.8978921350215125, 0.9227941176470589, 0.9289715789235122, 0.9382352941176471],
    [0.8830882352941176, 0.9073529411764706, 0.9198907502315908, 0.9455882352941176, 0.9588235294117647, 0.9588235294117647],
    [0.9198529411764705, 0.9448529411764706, 0.9639705882352941, 0.9727941176470588, 0.9772058823529411, 0.9779411764705882],
    [0.9299082672981759, 0.9562859204124259, 0.9707891235759292, 0.9727941176470588, 0.9289715789235122, 0.9819467862398598],
    [0.9389512509238021, 0.9618239085230987, 0.9725072109821821, 0.9772058823529411, 0.9830882352941176, 0.9819467862398598],
    [0.93988934117, 0.9618239085230987, 0.9778097321590823, 0.9815789127921371, 0.9831532526562236, 0.9830882352941176],
]
# for each value of max iterations, creating list of answers


learning_rate_perceptron = [0.05, 0.075, 0.1, 0.2, 0.5, 0.8]
max_iter_perceptron = [300, 500, 800, 1000, 1360]

learning_rate_logistic = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25]
max_epochs_logistic = [50, 100, 300, 500, 800, 1000]


for i in range(5):
    plt.plot(learning_rate_perceptron, perceptron_accuracies[i])
    plt.scatter(learning_rate_perceptron, perceptron_accuracies[i])
    plt.plot(learning_rate_perceptron, perceptron_train_accuracies[i])
    plt.scatter(learning_rate_perceptron, perceptron_train_accuracies[i])
    plt.title("Perceptron: Accuracy vs LR - Train And Test for %s examples"%max_iter_perceptron[i])
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0.65,0.91,0.01))
    plt.xticks(np.array([0.05, 0.075, 0.1, 0.2, 0.5, 0.8]), rotation = 80)
    # plt.show()
    plt.savefig('perceptron_'+str(i)+'.jpg')
    plt.clf()

for i in range(6):
    plt.plot(learning_rate_logistic, logistic_accuracies[i])
    plt.scatter(learning_rate_logistic, logistic_accuracies[i])
    plt.plot(learning_rate_logistic, logistic_train_accuracies[i])
    plt.scatter(learning_rate_logistic, logistic_train_accuracies[i])
    plt.title("Logistic Regression: Accuracy vs LR - Train and Test for %s iterations"%max_epochs_logistic[i])
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0.75,0.99,0.01))
    # plt.show()
    plt.savefig('logistic_'+str(i)+'.jpg')
    plt.clf()