# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries such as pandas, sklearn, matplotlib, and seaborn and Load the dataset HeightsWeights.csv.
2. Select the features Height (Inches) and Weight (Pounds).
3. Visualize the original data distribution using a scatter plot.
4. Apply StandardScaler to standardize the feature values and Apply PCA to reduce the dimensionality of the data.
5. Calculate the explained variance ratio of the principal components and Transform the dataset into principal components.
6. Visualize the PCA-transformed data using a scatter plot.

## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: POPURI SAHITHYA
RegisterNumber:  212225240106
*/
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('HeightsWeights.csv')
print(data.head())
X=data[['Height(Inches)','Weight(Pounds)']]
plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)',y='Weight(Pounds)',data=data)
plt.title("Original Data Distribution")
plt.show()
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)
print("Explained Variance Ratio:",pca.explained_variance_ratio_)
pca_df=pd.DataFrame(X_pca,columns=['PC1','PC2'])
plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1',y='PC2',data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

## Output:

<img width="627" height="148" alt="image" src="https://github.com/user-attachments/assets/932bbfd4-9867-499f-be49-e67f2c9f0322" />

<img width="696" height="592" alt="image" src="https://github.com/user-attachments/assets/e7adea22-f43d-403c-a58c-adcc084ebdcd" />

<img width="496" height="44" alt="image" src="https://github.com/user-attachments/assets/5fab3203-3bb0-433c-a919-8abc68807868" />

<img width="697" height="589" alt="image" src="https://github.com/user-attachments/assets/a19a51f6-e503-4654-bcc7-5f022197a37e" />

## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
