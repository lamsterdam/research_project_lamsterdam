from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mydata = pd.read_csv('iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length',
                                                      'petal_width', 'class'])


# splitting the data into an x and y where x contains the features and y the classifications
from sklearn.model_selection import train_test_split

x = mydata.iloc[:, :-1]
y = mydata.iloc[:, -1]

x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=0)

# converting them to four arrays
x_training = np.asarray(x_training)
x_testing = np.asarray(x_testing)
y_training = np.asarray(y_training)
y_testing = np.asarray(y_testing)

# calculating mean and standard deviation
x_training_mean = x_training.mean()
x_testing_mean = x_testing.mean()
x_training_standard_dev = x_training.std()
x_testing_standard_dev = x_testing.std()

# normalizing the data
x_training_normalize = (x_training - x_training_mean) / x_training_standard_dev
x_testing_normalize = (x_testing - x_testing_mean) / x_testing_standard_dev

# calculating the distance between each training point and the target point to be classified
def euclidean_distance(x_training, x_target):
    dist = []
    for i in range(0, len(x_training)):
        distance = 0
        training_point = x_training[i]
        for j in range(0, len(training_point)):
            distance += (training_point[j] - x_target[j]) ** 2
        distance = np.sqrt(distance)
        dist.append(distance)
        print(distance)
    dist = pd.DataFrame(data=dist, columns=['distance'])
    return dist

# finiding the nearbours to the target point for some value of k number of neighbours
def find_neighbours(point, k):
    neighbours = point.sort_values(by=['distance'], axis=0)
    neighbours = neighbours[:k]
    return neighbours

# finiding the most common classification for the closest neighbours and returning that classifier
def classify_neighbours(neighbours, y_training):
    counter = Counter(y_training[neighbours.index])
    classifier = counter.most_common()[0][0]
    return classifier

# implementing the function to classify the target point based on the closest neighbours the most common class of those neighbours
def knn(x_training, x_testing, y_training, k):
    classifier = []
    for x_target in x_testing:
        point = euclidean_distance(x_training, x_target)
        neighbours = find_neighbours(point, k)
        classifier_point = classify_neighbours(neighbours, y_training)
        classifier.append(classifier_point)
    return classifier


def main():
    # running algorithm with k = 3 as an example and building confusion matrix 
    k = 3
    classifier_test = knn(x_training_normalize, x_testing_normalize, y_training, k)
    
    from sklearn import metrics
    from sklearn.metrics import classification_report, confusion_matrix
    accuracy_matrix = confusion_matrix(y_testing, classifier_test)
    sns.heatmap(accuracy_matrix, annot=True, fmt="d")
    plt.title("KNN Confusion Matrix")
    plt.xlabel("Anticipated Classification")
    plt.ylabel("True Classification")
    print(classification_report(y_testing, classifier_test))
    plt.show()


if __name__ == "__main__":
    main()

