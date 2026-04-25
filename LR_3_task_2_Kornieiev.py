
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("Current working directory:", os.getcwd())

file_path = "Wholesale customers data.csv"


if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Файл '{file_path}' не знайдено. Перемісти його в робочу папку або вкажи повний шлях."
    )

data = pd.read_csv(file_path)
print(data.head())

print(data.describe())


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

print(pd.DataFrame(data_scaled).describe())

kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10)
kmeans.fit(data_scaled)

print("Inertia (k=2):", kmeans.inertia_)

SSE = []

for cluster in range(1, 20):
    kmeans = KMeans(
        n_clusters=cluster,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)


plt.figure(figsize=(10, 5))
plt.plot(range(1, 20), SSE, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method")
plt.show()


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
kmeans.fit(data_scaled)

pred = kmeans.predict(data_scaled)

frame = pd.DataFrame(data_scaled, columns=data.columns)
frame['cluster'] = pred

print(frame['cluster'].value_counts())
