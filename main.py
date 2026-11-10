
# ============================================================
# MACHINE LEARNING
# PENGELOMPOKAN MAKANAN BERDASARKAN KANDUNGAN NUTRISI
# METODE: K-MEANS CLUSTERING
# ============================================================

# %% 1. IMPORT LIBRARY

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# %% 2. MEMBUAT FOLDER HASIL

os.makedirs("hasil", exist_ok=True) 


# %% 3. MEMBACA DATASET

df = pd.read_csv("Nutrisi pangan Indonesia.csv")

print("===== DATASET AWAL =====")
print(df.head())

print("\nJumlah data:", len(df))
print("\nInformasi dataset:")
print(df.info())


# %% 4. PREPROCESSING DATA

fitur = [
    "calories",
    "proteins",
    "fat",
    "carbohydrate"
]

kolom_wajib = ["name"] + fitur

# Memastikan kolom tersedia
kolom_hilang = [
    kolom for kolom in kolom_wajib
    if kolom not in df.columns
]

if kolom_hilang:
    raise ValueError(
        f"Kolom berikut tidak ditemukan dalam dataset: {kolom_hilang}"
    )

# Mengubah fitur nutrisi menjadi angka
for kolom in fitur:
    df[kolom] = pd.to_numeric(
        df[kolom],
        errors="coerce"
    )

# Menghapus data kosong
df = df.dropna(subset=kolom_wajib)

# Menghapus data dengan nilai nutrisi negatif
df = df[(df[fitur] >= 0).all(axis=1)]

# Menghapus baris duplikat
df = df.drop_duplicates()

# Mengatur ulang index
df = df.reset_index(drop=True)

print("\n===== DATA SETELAH PREPROCESSING =====")
print(df.head())
print("\nJumlah data setelah preprocessing:", len(df))


# %% 5. MEMILIH FITUR NUTRISI

X = df[fitur]

print("\n===== FITUR YANG DIGUNAKAN =====")
print(X.head())


# %% 6. STANDARDISASI DATA

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n===== DATA SETELAH STANDARDISASI =====")
print(pd.DataFrame(X_scaled, columns=fitur).head())


# %% 7. MENENTUKAN JUMLAH CLUSTER DENGAN ELBOW METHOD

inertia = []
nilai_k = range(2, 9)

for k in nilai_k:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    list(nilai_k),
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Inertia")
plt.xticks(list(nilai_k))
plt.grid(True)

plt.savefig(
    "hasil/elbow_method.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# %% 8. EVALUASI SILHOUETTE SCORE

silhouette_scores = []

for k in nilai_k:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print(
        f"Jumlah Cluster: {k} | "
        f"Silhouette Score: {score:.4f}"
    )

# Memilih jumlah cluster dengan Silhouette Score tertinggi
k_optimal = list(nilai_k)[
    np.argmax(silhouette_scores)
]

print("\nJumlah cluster terbaik berdasarkan Silhouette Score:", k_optimal)


# %% 9. VISUALISASI SILHOUETTE SCORE

plt.figure(figsize=(8, 5))

plt.plot(
    list(nilai_k),
    silhouette_scores,
    marker="o"
)

plt.title("Evaluasi Silhouette Score")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(nilai_k))
plt.grid(True)

plt.savefig(
    "hasil/silhouette_score.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# %% 10. TRAINING MODEL K-MEANS

kmeans = KMeans(
    n_clusters=k_optimal,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

print("\n===== HASIL CLUSTERING =====")
print(df[["name", "cluster"]].head(20))


# %% 11. EVALUASI MODEL

score_akhir = silhouette_score(
    X_scaled,
    df["cluster"]
)

print("\n===== HASIL EVALUASI MODEL =====")
print("Jumlah Cluster:", k_optimal)
print("Silhouette Score:", round(score_akhir, 4))
print("Inertia:", round(kmeans.inertia_, 4))


# %% 12. ANALISIS JUMLAH MAKANAN SETIAP CLUSTER

jumlah_cluster = df["cluster"].value_counts().sort_index()

print("\n===== JUMLAH MAKANAN SETIAP CLUSTER =====")
print(jumlah_cluster)


# %% 13. ANALISIS RATA-RATA KANDUNGAN NUTRISI

rata_rata = df.groupby("cluster")[fitur].mean()

print("\n===== RATA-RATA NUTRISI SETIAP CLUSTER =====")
print(rata_rata.round(2))


# %% 14. VISUALISASI JUMLAH MAKANAN SETIAP CLUSTER

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="cluster"
)

plt.title("Jumlah Makanan pada Setiap Cluster")
plt.xlabel("Cluster")
plt.ylabel("Jumlah Makanan")

plt.savefig(
    "hasil/jumlah_cluster.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# %% 15. VISUALISASI RATA-RATA NUTRISI

rata_rata.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Rata-Rata Kandungan Nutrisi Setiap Cluster")
plt.xlabel("Cluster")
plt.ylabel("Nilai Nutrisi")
plt.xticks(rotation=0)
plt.legend(title="Kandungan Nutrisi")
plt.tight_layout()

plt.savefig(
    "hasil/rata_rata_nutrisi.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# %% 16. VISUALISASI CLUSTER MENGGUNAKAN PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="PCA1",
    y="PCA2",
    hue="cluster",
    palette="viridis",
    s=60
)

plt.title("Visualisasi Pengelompokan Makanan dengan K-Means")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster")

plt.savefig(
    "hasil/visualisasi_cluster.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# %% 17. MENAMPILKAN CONTOH MAKANAN SETIAP CLUSTER

print("\n===== CONTOH MAKANAN SETIAP CLUSTER =====")

for cluster in sorted(df["cluster"].unique()):

    print(f"\nCluster {cluster}:")

    makanan = df[df["cluster"] == cluster]["name"]

    for nama in makanan.head(10):
        print("-", nama)


# %% 18. MENYIMPAN HASIL CLUSTERING

df.to_csv(
    "hasil/hasil_clustering.csv",
    index=False
)

print("\nHasil clustering berhasil disimpan di:")
print("hasil/hasil_clustering.csv")


# %% 19. KESIMPULAN

print("\n==========================================")
print("              KESIMPULAN")
print("==========================================")

print(
    f"Program berhasil mengelompokkan makanan "
    f"menjadi {k_optimal} cluster."
)

print(
    f"Nilai Silhouette Score yang diperoleh adalah "
    f"{score_akhir:.4f}."
)

print(
    "Pengelompokan dilakukan berdasarkan kandungan "
    "kalori, protein, lemak, dan karbohidrat."
)

print(
    "Setiap cluster menunjukkan kelompok makanan "
    "dengan karakteristik kandungan nutrisi yang serupa."
)

print("\nProgram selesai dijalankan.")
