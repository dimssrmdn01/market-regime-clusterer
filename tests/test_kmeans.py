import sys
import os
import numpy as np

# Menyisipkan direktori utama (root) ke dalam sistem path Python agar bisa terbaca
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_clusterer import CustomKMeans

def test_kmeans_initialization():
    """Menguji apakah parameter model diinisialisasi dengan benar."""
    model = CustomKMeans(k=4, max_iters=50)
    assert model.k == 4
    assert model.max_iters == 50

def test_kmeans_fit_shape():
    """Menguji apakah algoritma K-Means mengembalikan dimensi label yang sesuai dengan jumlah data."""
    np.random.seed(42)
    X_dummy = np.random.rand(50, 2)
    
    model = CustomKMeans(k=3)
    labels, iters = model.fit(X_dummy)
    
    assert len(labels) == 50, "Jumlah label harus sama dengan jumlah baris data."
    assert len(np.unique(labels)) <= 3, "Jumlah klaster unik tidak boleh melebihi nilai K."
    assert iters > 0, "Iterasi harus berjalan minimal 1 kali."

def test_inertia_calculation():
    """Menguji apakah skor inersia dihitung dengan tipe data yang benar."""
    X_dummy = np.random.rand(20, 2)
    model = CustomKMeans(k=2)
    labels, _ = model.fit(X_dummy)
    
    inertia = model.calculate_inertia(X_dummy, labels)
    
    assert isinstance(inertia, float), "Skor inersia harus berupa angka desimal (float)."
    assert inertia >= 0, "Skor inersia tidak boleh bernilai negatif."