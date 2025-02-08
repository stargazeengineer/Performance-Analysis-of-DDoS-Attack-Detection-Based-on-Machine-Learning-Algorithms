# Performance Analysis of DDoS Attack Detection Based on Machine Learning Algorithms

近年來，分散式阻斷服務（DDoS）攻擊已成為網路安全領域的一大挑戰。本研究旨在分析不同的機器學習演算法，並找出最佳方法以提高 DDoS 攻擊檢測的準確性。我選擇了五種經典的機器學習演算法：隨機森林（Random Forest）、支援向量機（SVM）、K-近鄰（KNN）、邏輯回歸（Logistic Regression）以及朴素貝葉斯（Naive Bayes），並在 CIC-DDoS2019 資料集上進行實驗。

結果顯示，隨機森林、SVM 和 KNN 的表現最佳，準確率與 F1-score 均達到 99.74% 以上。邏輯回歸的準確率為 97.82%，但仍存在少量分類錯誤。由於朴素貝葉斯假設特徵之間相互獨立，因此在處理網路流量數據時表現較差。我還使用 ROC 曲線與 AUC 值來評估各模型的效能。

本研究分析了不同機器學習演算法在 DDoS 攻擊檢測中的適用性及其優缺點，為實際應用提供了參考。本專案的程式碼包含完整的數據預處理、模型訓練與評估過程，使研究人員能夠輕鬆重現實驗結果。該程式碼也包含各步驟的詳細說明。

# What is Machine Learning

# Preprocessing

![image](https://github.com/user-attachments/assets/04b87d3d-c219-49e7-8c00-30cede289100)

# Random Forest


