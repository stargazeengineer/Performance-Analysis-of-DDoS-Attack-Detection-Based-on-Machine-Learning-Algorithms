# Performance Analysis of DDoS Attack Detection Based on Machine Learning Algorithms

In recent years, Distributed Denial of Service (DDoS) attacks have become a major challenge in the field of cybersecurity. This study aims to analyze various machine learning algorithms and identify the most effective method for improving the accuracy of DDoS attack detection. I selected five classic machine learning algorithms—Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Logistic Regression, and Naive Bayes—and conducted experiments using the CIC-DDoS2019 dataset.

The results show that Random Forest, SVM, and KNN performed the best, achieving accuracy and F1-scores of over 99.74%. Logistic Regression achieved an accuracy of 97.82%, but still had a small number of misclassifications. Due to its assumption of feature independence, Naive Bayes performed poorly when processing network traffic data. I also used ROC curves and AUC values to evaluate the performance of each model. Finally, a lightweight DDoS defense system was developed based on the model trained with the Random Forest algorithm, designed for local deployment.

This study analyzes the applicability, advantages, and disadvantages of different machine learning algorithms in DDoS attack detection, providing a valuable reference for real-world applications. The project code includes complete data preprocessing, model training, and evaluation procedures, enabling researchers to easily reproduce the experimental results. Detailed explanations for each step are also included in the code.

# What is Machine Learning

# About CIC-DDoS2019 Dataset

# Preprocessing

![image](https://github.com/user-attachments/assets/04b87d3d-c219-49e7-8c00-30cede289100)

# Random Forest

# SVM

# KNN

# Logic Regressor

# Naive Bayes

# Conclusion
