import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import sqlite3
from typing import Tuple

# Setup konfigurasi halaman Streamlit
st.set_page_config(page_title="Market Volatility Clustering Engine", layout="wide")

class DatabaseManager:
    """Mengatur operasi penyimpanan metrik stabilitas klaster ke dalam SQLite database."""
    
    def __init__(self, db_name: str = "market_regimes.db"):
        self.db_name = db_name

    def init_db(self) -> None:
        """Inisialisasi tabel database relasional jika belum terbentuk."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cluster_stability_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                k_clusters INTEGER,
                inertia_score REAL,
                iterations_converged INTEGER
            )
        ''')
        conn.commit()
        conn.close()

    def log_stability_metrics(self, k: int, inertia: float, iterations: int) -> None:
        """Menyimpan metrik performa algoritma ke dalam database."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO cluster_stability_logs (timestamp, k_clusters, inertia_score, iterations_converged)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, k, float(inertia), int(iterations)))
        conn.commit()
        conn.close()

    def fetch_latest_logs(self, limit: int = 5) -> pd.DataFrame:
        """Mengambil histori log audit terbaru dari tabel database."""
        conn = sqlite3.connect(self.db_name)
        query = f"SELECT * FROM cluster_stability_logs ORDER BY id DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df


class CustomKMeans:
    """Implementasi algoritma K-Means Clustering dari dasar menggunakan matriks NumPy."""
    
    def __init__(self, k: int = 3, max_iters: int = 100, tol: float = 1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        
    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, int]:
        """Proses iterasi optimasi penempatan nilai centroid."""
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]
        
        for i in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            old_centroids = self.centroids.copy()
            for cluster_idx in range(self.k):
                cluster_points = X[labels == cluster_idx]
                if len(cluster_points) > 0:
                    self.centroids[cluster_idx] = cluster_points.mean(axis=0)
            
            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                return labels, i + 1
                
        return labels, self.max_iters

    def calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Menghitung metrik stabilitas Within-Cluster Sum of Squares."""
        inertia = 0.0
        for cluster_idx in range(self.k):
            cluster_points = X[labels == cluster_idx]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[cluster_idx]) ** 2)
        return float(inertia)


@st.cache_data
def generate_market_data() -> pd.DataFrame:
    """Generasi dataset simulasi pergerakan imbal hasil dan volatilitas pasar."""
    np.random.seed(42)
    regime_1 = np.random.normal(loc=[0.02, 0.12], scale=[0.01, 0.02], size=(150, 2))  # Sideways
    regime_2 = np.random.normal(loc=[0.06, 0.28], scale=[0.02, 0.04], size=(150, 2))  # Bull Run
    regime_3 = np.random.normal(loc=[-0.09, 0.45], scale=[0.04, 0.06], size=(150, 2)) # Crash Zone
    
    data = np.vstack([regime_1, regime_2, regime_3])
    df = pd.DataFrame(data, columns=['Daily_Returns', 'Rolling_Volatility'])
    return df

# Inisialisasi Database
db = DatabaseManager()
db.init_db()

# Render Komponen Judul Aplikasi
st.title("Market Volatility Clustering Engine")
st.subheader("Analisis Segmentasi Regime Pasar Menggunakan K-Means Pure NumPy dan Sinkronisasi SQL")
st.write("---")

# Panel Kontrol Utama pada Sidebar
st.sidebar.header("Konfigurasi Parameter Algoritma")
k_clusters = st.sidebar.slider("Jumlah Klaster (K)", min_value=2, max_value=5, value=3)
max_iterations = st.sidebar.slider("Maksimum Iterasi", min_value=10, max_value=200, value=100)

# Proses Ingesti Data Simulasi Pasar
df_market = generate_market_data()
X_matrix = df_market[['Daily_Returns', 'Rolling_Volatility']].values

# Eksekusi Pipeline Matematika K-Means
model = CustomKMeans(k=k_clusters, max_iters=max_iterations)
labels, iterations_run = model.fit(X_matrix)
inertia_val = model.calculate_inertia(X_matrix, labels)

df_market['Cluster'] = labels.astype(str)

# Layout Tampilan Metrik Evaluasi Utama
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric(label="Klaster K Aktif", value=f"{k_clusters} Regimes")
with col_m2:
    st.metric(label="Iterasi Hingga Konvergen", value=f"{iterations_run} Siklus")
with col_m3:
    st.metric(label="Skor Inertia Penyelarasan", value=f"{round(inertia_val, 4)}")

st.write("---")

col_left, col_right = st.columns([1.8, 1.2])

with col_left:
    st.markdown("### Visualisasi Spatial Analisis Regime Pasar")
    fig = px.scatter(df_market, x='Daily_Returns', y='Rolling_Volatility', color='Cluster',
                     title="Segmentasi Volatilitas Menggunakan K-Means Berbasis NumPy",
                     labels={'Daily_Returns': 'Daily Returns', 'Rolling_Volatility': 'Rolling Volatility'},
                     color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("### Manajemen Kepatuhan Data Model")
    st.write("Simpan dan audit parameter performa model penempatan klaster pasar ke dalam database relasional lokal.")
    
    if st.button("Simpan Metrik Stabilitas ke SQL"):
        db.log_stability_metrics(k_clusters, inertia_val, iterations_run)
        st.success("Performa model berhasil dikunci ke dalam tabel cluster_stability_logs.")

# Tampilan Tabel Audit Historis di Bagian Dasar Halaman
st.write("---")
st.markdown("### Audit Log Histori Performa Model (Live Query SQL Database)")
df_historical_logs = db.fetch_latest_logs(limit=5)

if not df_historical_logs.empty:
    st.dataframe(df_historical_logs, use_container_width=True)
else:
    st.info("Database rekor audit internal masih kosong. Gunakan tombol diatas untuk mengirim entri pertama.")