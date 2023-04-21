# Research Paper: Report on Analysis of K-Nearest Neighbours Algorithm
* **Name**: Leigh-Riane Amsterdam
* **Semester**: Spring 2023
* **Topic**: K - Nearest Neighbours
* **Link** The Repository: https://github.com/lamsterdam/research_project_lamsterdam.git

## Introduction

This research report focuses on the analysis of the K-Nearest Neighbours algorithm and explores the accuracy of the algorithm by using a Confusion Matrix. This algorithm is used in the area of Machine Learning to make classifiactions and predictions based on the proximity of data points to one another. It is able to solve the problem of assigning a classification to unclassified points based on the distance between the point and other known points when the overall distribution of the data is unknown or hard to determine. This is because the algorithm does not take into account the distribution of the data points and thus it does not need this knowledge in order to run successfully. 
The K-Nearest Neighbours algorithm was first introduced in an unpublished US Air Force School of Aviation Medicine report by Evelyn Fix and Joseph Hodges in 1951. They were both staticians at Berkley at the time and the results of their report later became known as the kNN algorithm. 
In remainder of this report I will examine the implementation of the K-Nearest Neighbours algorithm in python and will analyze the performance of the algorithm based on the pre-calissfied iris dataset.  


## Analysis of Algorithm/Datastructure

| Algorithm |  Time | Space | 
| :-- | :-- |  :-- | 
| K-Nearest Neighbours | $O(nlogn)$ | $O(1)$ | 


The table above illustrates the time and space complexity for this algorithm. As shown, the time complexity is $O(nlogn)$ and the space complexity is $O(1)$. 

- General analysis of the algorithm/datastructure

The K-Nearest neighbours alogorithm is known as a supervised machine learning algorithm. This is because the way in which an algorithm learns how to classify points from the training data set is similar to the way in which a teacher supervises her students' learning. The algorithm learns by making decisions on the classifications, and the users already knowing the answers can correct the algorithm when needed. It uses the Euclidean Distance formula to calculate the distance between the unclassified target point and several other classified training points, and from these calculations is able to classify the target point based on proximity. The mathematical formula for Euclidean Distance is illustrated below: 


## Empirical Analysis

For this analysis I wrote several function to implement the K-Nearest Neighbours algorithm in python and used it to classify the iris data set. Since this data is already pre-classified, my goal was to test how accurate my implementation was at classifiying the species of flower based on the features. I split the data into two groups such that 70% of the data was used for training point data and 30% for testing point data. Based on the results of the model I compared it to what the true classification of the points should be to determine how accurate the model was. This was done using a Confusion Matrix where I could further analyze the accuracy, precision, recall and f1-score. In addition, I plotted a heatmap representation of the matrix to illustrate this visually as well.  


## Application

This algorithm is used for classification of data points based on their proximity to other data points. Based on the most common classification of nearby data points, the target data point is then assigned a classification or grouping that matches the majority vote of nearby classifications. Some areas where this algorithm is applied is in finance to determine the risk associated with loaning money to individuals based on their credit score, in healthcare to make predictions on heart attack likelihood and in pattern recoginition to classify handwritten text such as numbers. The algorithm is useful in each of these areas because for example in finance, it can look at the credit score of past loan applicants to the bank and based on the behaviours of those persons can classify future loan applicants to a certain risk based on the nearness of their credit score to others before them. In a similar way it, within healthcare the algorithm can make a calculation on the connection between suffering from a heart attack and a patient's gene expressions using the proximity to other patients gene expression calculation and whether they suffered from a heart attack. Finally, in pattern recoginition it is also useful for classifying handwritten numbers by determining the proximity to other classified handwritten numbers and patterns, and assigning a number to the handwritte number based on the most common nearby number. 


## Implementation

For this implementation I used python as well as the pre-classified iris dataset which was obtained from the UCI Machine Learning Repository Laboratory. I imported the dataset to python and then split the data into my training sample and my testing sample to run the algorithm on. The Pandas, NumPy, matplotlib and seaborn libraries were also inported for this project. 

Firstly, I imported the data set into my file then divided it into a dataframe where x contained the features used forclassification and y contained the three different flower species classification; Iris-sertosa, Iris-versicolor and Iris-verginica. Then using train_test_split() function I split the data into testing data and training data, at 70% and 30% respectively. 

```python
# splitting the data into an x and y where x contains the features and y the classifications
from sklearn.model_selection import train_test_split

x = mydata.iloc[:, :-1]
y = mydata.iloc[:, -1]

x_training, x_testing, y_training, y_testing = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=0)
```

After the splitting was complete and verfied to have a training data size of 105 points and testing data size of 45 points, I calculated the mean and standard deviation to then normalize the data. The equation I used for normalization is illustrated as follows:


```python
# calculating mean and standard deviation
x_training_mean = x_training.mean()
x_testing_mean = x_testing.mean()
x_training_standard_dev = x_training.std()
x_testing_standard_dev = x_testing.std()

# normalizing the data
x_training_normalize = (x_training - x_training_mean) / x_training_standard_dev
x_testing_normalize = (x_testing - x_testing_mean) / x_testing_standard_dev
```

Next, I calculated the euclidean distance between each point in the training data set to the target point that needed to be classified and added it to my table of distances. These distances were then sorted and used to find the nearest neighbours of the target point based on the value of k. This meant that is k as equal to 3, then of the sorted distances the first three neighbouring points with the smallest distance would be returned. Of this subset of neighbours, their classifications in terms of the species of flower that they belonged to was obtained and the most common species among the neighbours was specified as the classification of that target point. 

```python
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
```

Finally, I wrote the function to return the predicted species classifier for all of the points in the testing data set, based on the neighbouring point's classifications and their distances. In main(), I specified the value of k and also constructed the confusion matrix and a heatmap that illustrated the anticipated classification and the true classification that was expected.

```python
def knn(x_training, x_testing, y_training, k):
    classifier = []
    for x_target in x_testing:
        point = euclidean_distance(x_training, x_target)
        neighbours = find_neighbours(point, k)
        classifier_point = classify_neighbours(neighbours, y_training)
        classifier.append(classifier_point)
    return classifier

```

```python
def main():
    k = 3
    classifier_test = knn(x_training_normalize, x_testing_normalize, y_training, k)
    print(classifier_test)

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

```

For my implementation I did use several sources for reference when building out the project, these included the "Implementation of K-Nearest Neighbours" on GeeksforGeeks, "Machine Learning - Confusion Matrix" on W3School. All of these are also mentioned under the References section of this report with additional details.
Some of the challenges I faces were understanding how the algorithm worked in the beginning and how to implement it. I was not well acqainted with any Machine Learning algorithms prior to this research project, so it took some time to unpack what the algorithm was accomplishing and how it could be applied. Another challenging aspect was learning how to use the libraries and data frames as I was not familiar with numpy, pandas or any of the additional libraries. The third challenging part was analyzing the results and undertanding what the confusion matrix was showing me. The resulting matrix was a 3X3 matrix because I used three different species as the classifications, but I was used to finding 2X2 matrices online. Therefore, it took some time to understand how to interpret the results and ensure that I was on the right track. 



## Summary

![Alt text](https://github.com/lamsterdam/research_project_lamsterdam/blob/main/confusion.png)

In summary, the model was mostly accurate as illustrated in the image above. The precision of accurately classifying the flowers ranged from 0.92 to 1.00 and the accuracy of the model was 98%. I also constructed a heatmap to show the confusion matrix which is shown in the image below. The model made a slight mistake when it came to Iris-virginica but for the other two types it was able to accuately make the classification.

![Alt text](https://github.com/lamsterdam/research_project_lamsterdam/blob/main/confusion_matrix_map.png)

Overall, prior to writing this research paper I had very little knowledge of machine learning algorithms, as well as the libraries in python. Therefore, I learnt about supervised machine learning algorithms and how to construct one, and I also gained a basic understanding of several libraries such as numpy and pandas.  

## References
1. GeeksforGeeks, "K-Nearest neighbours", 14 March, 2023. https://www.geeksforgeeks.org/k-nearest-neighbours/
2. Scholarpedia, "K-nearest neighbor, 3 November, 2013. http://www.scholarpedia.org/article/K-nearest_neighbor
3. GeeksforGeeks, "Implementation of K-Nearest Neighbours", 9 November, 2022. https://www.geeksforgeeks.org/implementation-k-nearest-neighbors/
4. W3School, "Machine Learning - Confusion Matrix", https://www.w3schools.com/python/python_ml_confusion_matrix.asp
5. IBM, "What is the k-nearest neighbors algorithm?", https://www.ibm.com/topics/knn
6. ResearchGate, "Improved Handwritten Digit Recognition using Quantum K-Nearest Neighbor Algorithm", July 2019, https://www.researchgate.net/publication/332880911_Improved_Handwritten_Digit_Recognition_using_Quantum_K-Nearest_Neighbor_Algorithm
7. IOPScience, "Credit scoring analysis using weighted k-nearest neighbor - Journal of Physics: Conference Series", https://iopscience.iop.org/article/10.1088/1742-6596/1025/1/012114