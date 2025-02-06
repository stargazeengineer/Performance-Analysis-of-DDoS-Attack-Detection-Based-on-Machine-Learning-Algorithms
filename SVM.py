import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 載入資料集
df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# 處理非數值型欄位並刪除不需要的欄位
df = df.drop(columns=['Flow ID', ' Source IP', ' Destination IP', ' Timestamp'])

# 將 ' Label' 欄位轉換為數值型
df[' Label'] = df[' Label'].astype('object')
df.loc[df[' Label'] == 'BENIGN', ' Label'] = 0  # BENIGN to 0
df.loc[df[' Label'] == 'DDoS', ' Label'] = 1    # DDoS to 1 (or any other value != 0)
df[' Label'] = df[' Label'].astype(dtype=int)

# 處理缺失值和無窮大值
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# 分割特徵（X）和標籤（y）
def xs_y(df_, targ):
    if not isinstance(targ, list):
        xs = df_[df_.columns.difference([targ])].copy()
    else:
        xs = df_[df_.columns.difference(targ)].copy()
    y = df_[targ].copy()
    return xs, y

target = ' Label'
X, y = xs_y(df, target)

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 特徵縮放 (SVM 對於特徵縮放敏感)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 初始化 SVM 模型
svm_model = SVC(random_state=42)  # You can adjust kernel, C, gamma, etc.

# 訓練 SVM 模型
svm_model.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = svm_model.predict(X_test)

# 評估模型
print("----- 測試資料集的評估 -----")
print("測試資料集準確度:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 混淆矩陣繪製
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'DDoS'], yticklabels=['BENIGN', 'DDoS'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
