import os
import numpy as np
import pandas as pd
import sqlite3
from typing import Tuple, List

class DatabaseManager:
    """Mengatur semua operasi pembentukan dan penyimpanan data ke SQLite database."""
    
    def __init__(self, db_name: str = "market_regimes.db"):
        self.db_name = db_name

    def init_db(self) -> None:
        """Inisialisasi skema tabel database untuk mencatat stabilitas klaster."""
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
        """Menyimpan metrik evaluasi model secara permanen ke dalam database."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO cluster_stability_logs (timestamp, k_clusters, inertia_score, iterations_converged)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, k, float(inertia), int(iterations)))
        conn.commit()
        conn.close()
        print("[INFO][SQL_LOG] Metrik stabilitas klaster berhasil disimpan ke database.")

    def fetch_latest_logs(self, limit: int = 5) -> pd.DataFrame:
        """Mengambil data log terbaru dari database untuk proses audit."""
        conn = sqlite3.connect(self.db_name)
        query = f"SELECT * FROM cluster_stability_logs ORDER BY id DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df


class CustomKMeans:
    """Implementasi algoritma K-Means Clustering menggunakan operasi matriks NumPy."""
    
    def __init__(self, k: int = 3, max_iters: int = 100, tol: float = 1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        
    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, int]:
        """Menjalankan proses optimasi centroid secara iteratif."""
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]
        
        for i in range(self.max_iters):
            # Hitung jarak Euclidean dari setiap titik ke seluruh centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            
            # Penetapan klaster berdasarkan jarak terdekat
            labels = np.argmin(distances, axis=1)
            
            # Pembaruan posisi centroid berdasarkan rata-rata nilai anggota klaster
            old_centroids = self.centroids.copy()
            for cluster_idx in range(self.k):
                cluster_points = X[labels == cluster_idx]
                if len(cluster_points) > 0:
                    self.centroids[cluster_idx] = cluster_points.mean(axis=0)
            
            # Evaluasi konvergensi berdasarkan ambang batas toleransi
            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                return labels, i + 1
                
        return labels, self.max_iters

    def calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Menghitung nilai Within-Cluster Sum of Squares (Inertia) secara manual."""
        inertia = 0.0
        for cluster_idx in range(self.k):
            cluster_points = X[labels == cluster_idx]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[cluster_idx]) ** 2)
        return float(inertia)


def generate_market_data() -> np.ndarray:
    """Generasi data sintetis untuk simulasi indikator volatilitas pasar."""
    np.random.seed(42)
    regime_1 = np.random.normal(loc=[0.05, 0.1], scale=[0.02, 0.02], size=(100, 2))  # Low Volatility Bullish
    regime_2 = np.random.normal(loc=[-0.08, 0.4], scale=[0.04, 0.05], size=(100, 2)) # High Volatility Bearish
    regime_3 = np.random.normal(loc=[0.01, 0.22], scale=[0.01, 0.03], size=(100, 2)) # Mid Volatility Sideways
    return np.vstack([regime_1, regime_2, regime_3])


if __name__ == "__main__":
    db = DatabaseManager()
    db.init_db()
    
    print("[PROCESS] Memulai eksekusi analisis segmentasi regime pasar...")
    X_market = generate_market_data()
    
    k_target = 3
    model = CustomKMeans(k=k_target, max_iters=50)
    labels, iterations = model.fit(X_market)
    inertia_score = model.calculate_inertia(X_market, labels)
    
    # Amankan hasil ke dalam database log
    db.log_stability_metrics(k_target, inertia_score, iterations)
    
    print("\n============================================================")
    print("LAPORAN EVALUASI MODEL")
    print("============================================================")
    print(f"Jumlah Klaster (K)       : {k_target}")
    print(f"Iterasi Konvergensi      : {iterations}")
    print(f"Skor Keselarasan Inertia : {round(inertia_score, 4)}")
    
    print("\n============================================================")
    print("AUDIT LOG DATABASE - 5 DATA TERBARU")
    print("============================================================")
    df_logs = db.fetch_latest_logs(limit=5)
    print(df_logs.to_string(index=False))