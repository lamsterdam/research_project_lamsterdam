# Research Paper: Report on Analysis of K-Nearest Neighbours Algorithm
* **Name**: Leigh-Riane Amsterdam
* **Semester**: Spring 2023
* **Topic**: K - Nearest Neighbours
* **Link** The Repository: https://github.com/lamsterdam/research_project_lamsterdam.git

## Introduction
- What is the algorithm/datastructure?
- What is the problem it solves? 
- Provide a brief history of the algorithm/datastructure. (make sure to cite sources)
- Provide an introduction to the rest of the paper. 

This research report focuses on the analysis of the K-Nearest Neighbours algorithm and explores how the accuracy of the algorithm changes based on data sizes and values of k.
The algorithm is used in the area of Machine Learning to make classifiactions and predictions based on the proximity of data points to one another. Thus, this algorithm is able to solve the problem of assigning a classification to unclassified points based on the distance between the point and other known points when the overall distribution of the data is unknown or hard to determine. This is because the algorithm does not take into account the distribution of the data points and thus it does not need this knowledge in order to run successfully. 
The K-Nearest Neighbours algorithm was first introduced in an unpublished US Air Force School of Aviation Medicine report by Evelyn Fix and Joseph Hodges in 1951. They were both staticians at Berkley at the time and the results of their report later became known as the kNN algorithm. 
In remainder of this report I will examine the implementation of the K-Nearest Neighbours algorithm in python and will analyze the performance of the algorithm as the size of the training data set increases for odd and even values of k. 


## Analysis of Algorithm/Datastructure
Make sure to include the following:
- Time Complexity
- Space Complexity

| Algorithm |  Time | Space | 
| :-- | :-- |  :-- | 
| K-Nearest Neighbours | $O(nlogn)$ | $O(1)$ | 


The table above illustrates the time and space complexity for this algorithm. As shown, the time complexity is $O(nlogn)$ and the space complexity is $O(1)$. 

- General analysis of the algorithm/datastructure

It uses the Euclidean Distance formula to calculate the distance between the unclassified target point and several other classified training points, and from these calculations is able to classify the target point based on proximity.

## Empirical Analysis
- What is the empirical analysis?
- Provide specific examples / data.


## Application
- What is the algorithm/datastructure used for?
- Provide specific examples
- Why is it useful / used in that field area?
- Make sure to provide sources for your information.


## Implementation
- What language did you use?
- What libraries did you use?
- What were the challenges you faced?
- Provide key points of the algorithm/datastructure implementation, discuss the code.
- If you found code in another language, and then implemented in your own language that is fine - but make sure to document that.


## Summary
- Provide a summary of your findings
- What did you learn?


## References
1. GeeksforGeeks, "K-Nearest neighbours", 14 March, 2023. https://www.geeksforgeeks.org/k-nearest-neighbours/
2. Scholarpedia, "K-nearest neighbor, 3 November, 2013. http://www.scholarpedia.org/article/K-nearest_neighbor