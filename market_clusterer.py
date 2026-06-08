import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

<<<<<<< HEAD
# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisis Volatilitas Pasar", layout="wide")
st.title("📈 Mesin Analisis Rezim & Peramalan Tren Pasar Finansial")
st.markdown("Dashboard *unsupervised machine learning* dan peramalan kuantitatif murni dari *scratch* (NumPy murni, tanpa scikit-learn).")
st.divider()

# --- 1. UNDUH DATA PASAR DENGAN SISTEM CADANGAN (AUTOMATIC FALLBACK) ---
=======
#KONFIGURASI HALAMAN 
st.set_page_config(page_title="Analisis Volatilitas Pasar", layout="wide")
st.title("📈 Mesin Klasterisasi Volatilitas Pasar (K-Means Murni)")
st.markdown("Dashboard *machine learning* tanpa pengawasan (*unsupervised*) untuk mengelompokkan rezim pasar berdasarkan tingkat keuntungan dan volatilitas.")
st.divider()

#1. UNDUH DATA PASAR DENGAN SISTEM CADANGAN (AUTOMATIC FALLBACK)
>>>>>>> 4bc216023ecf75d203eb237788c8c04b73b073cf
@st.cache_data
def fetch_real_market_data(ticker, period="1y"):
    """Mengambil data asli dari Yahoo Finance dengan sistem cadangan simulasi jika koneksi diblokir."""
    try:
        import requests
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        asset = yf.Ticker(ticker, session=session)
        data = asset.history(period=period)
        
<<<<<<< HEAD
=======
        #JIKA BERHASIL TERHUBUNG KE YAHOO FINANCE
>>>>>>> 4bc216023ecf75d203eb237788c8c04b73b073cf
        if not data.empty:
            df = pd.DataFrame()
            df['Close'] = data['Close'].values
            df['Daily_Return'] = df['Close'].pct_change() * 100
            df['Rolling_Vol'] = df['Daily_Return'].rolling(window=5, min_periods=1).std().fillna(0)
            df['Tanggal'] = pd.to_datetime(data.index).tz_localize(None)
            df['Hari'] = np.arange(1, len(df) + 1)
            df.dropna(inplace=True)
            return df, "🟢 Berhasil Terhubung: Menampilkan Data Riil Secara Langsung dari Yahoo Finance API"
            
    except Exception:
        pass
        
<<<<<<< HEAD
    # --- RENCANA CADANGAN: JIKA YAHOO MEMBLOKIR (SIMULASI STOKASTIK) ---
    np.random.seed(42)
    n_days = 250 if period == "1y" else (125 if period == "6m" else 500)
    
=======
    #RENCANA CADANGAN: JIKA YAHOO MEMBLOKIR (SIMULASI STOKASTIK REPRISAL) 
    np.random.seed(42)
    n_days = 250 if period == "1y" else (125 if period == "6m" else 500)
    
    #Karakteristik volatilitas & harga awal tiap aset
>>>>>>> 4bc216023ecf75d203eb237788c8c04b73b073cf
    if "BTC" in ticker:
        start_price, mu, sigma, label = 65000, 0.05, 3.5, "Bitcoin (BTC-USD)"
    elif "ETH" in ticker:
        start_price, mu, sigma, label = 3200, 0.04, 4.0, "Ethereum (ETH-USD)"
    elif "GC=F" in ticker:
        start_price, mu, sigma, label = 2300, 0.01, 0.9, "Emas (GC=F)"
    elif "AAPL" in ticker:
        start_price, mu, sigma, label = 175, 0.02, 1.5, "Saham Apple (AAPL)"
    else:
        start_price, mu, sigma, label = 5100, 0.02, 1.1, "Indeks S&P 500 (^GSPC)"
        
<<<<<<< HEAD
=======
    #Pembuatan data tiruan menggunakan Geometric Brownian Motion
>>>>>>> 4bc216023ecf75d203eb237788c8c04b73b073cf
    returns = np.random.normal(loc=mu/n_days, scale=sigma/100, size=n_days)
    price_history = [start_price]
    for r in returns:
        price_history.append(price_history[-1] * (1 + r))
        
    date_range = pd.date_range(end=pd.Timestamp.now(), periods=n_days + 1, freq='B')
    
    df = pd.DataFrame({
        'Tanggal': date_range,
        'Close': price_history
    })
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Rolling_Vol'] = df['Daily_Return'].rolling(window=5, min_periods=1).std().fillna(0)
    df['Hari'] = np.arange(1, len(df) + 1)
    df.dropna(inplace=True)
    
    return df, f"⚠️ Koneksi Yahoo Finance Terbatasi/RTO. Mengaktifkan Mesin Simulasi Stokastik {label} Secara Otomatis."

<<<<<<< HEAD
# --- 2. ALGORITMA K-MEANS BUATAN SENDIRI (FROM SCRATCH) ---
=======
#2. ALGORITMA K-MEANS BUATAN SENDIRI (FROM SCRATCH) 
>>>>>>> 4bc216023ecf75d203eb237788c8c04b73b073cf
def custom_kmeans(X, k, max_iters=100):
    """Implementasi K-Means Clustering menggunakan matematika murni NumPy."""
    n_samples = X.shape[0]
    np.random.seed(42)
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    
    for _ in range(max_iters):
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids) ** 2).sum(axis=2))
        cluster_labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([X[cluster_labels == j].mean(axis=0) if len(X[cluster_labels == j]) > 0 else centroids[j] for j in range(k)])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return cluster_labels, centroids

<<<<<<< HEAD
# --- 3. FUNGSI PERAMALAN TREN MURNI NUMPY (REGRESI LINEAR OLS) ---
def hitung_tren_masa_depan(df, hari_ke_depan=30):
    """Menghitung proyeksi tren harga ke depan menggunakan rumus matematika murni."""
    X = df['Hari'].values
    y = df['Close'].values
    n = len(X)
    
    # Rumus Linear Regression Slope (m) & Intercept (c)
    m = (n * np.sum(X * y) - np.sum(X) * np.sum(y)) / (n * np.sum(X**2) - (np.sum(X))**2)
    c = (np.sum(y) - m * np.sum(X)) / n
    
    # Proyeksi indeks hari dan harga masa depan
    hari_mendatang = np.arange(X[-1] + 1, X[-1] + 1 + hari_ke_depan)
    harga_proyeksi = m * hari_mendatang + c
    
    # Pembuatan komponen tanggal proyeksi baru (Business Days)
    tanggal_terakhir = df['Tanggal'].max()
    tanggal_mendatang = pd.date_range(start=tanggal_terakhir + pd.Timedelta(days=1), periods=hari_ke_depan, freq='B')
    
    df_proyeksi = pd.DataFrame({
        'Tanggal': tanggal_mendatang,
        'Proyeksi Harga': harga_proyeksi
    })
    return df_proyeksi, m

# --- 4. SIDEBAR: KONTROL PARAMETER & PILIHAN ASET ---
st.sidebar.header("🎛️ Kontrol Pasar & Model")
=======
#3. SIDEBAR: KONTROL PARAMETER & PILIHAN ASET 
st.sidebar.header("🎛️ Kontrol Pasar & Model")

#Menu Pilihan Aset
>>>>>>> 4bc216023ecf75d203eb237788c8c04b73b073cf
asset_mapping = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Emas (GC=F)": "GC=F",
    "Saham Apple (AAPL)": "AAPL",
    "Indeks S&P 500 (^GSPC)": "^GSPC"
}
asset_choice = st.sidebar.selectbox("Pilih Aset Pasar", list(asset_mapping.keys()))
ticker_symbol = asset_mapping[asset_choice]
<<<<<<< HEAD
time_period = st.sidebar.selectbox("Rentang Waktu Histori", ["6m", "1y", "2y"], index=1)

st.sidebar.divider()
st.sidebar.subheader("🎯 Parameter Klasterisasi")
k_clusters = st.sidebar.slider("Jumlah Klaster (K)", min_value=2, max_value=5, value=3)
max_iter_input = st.sidebar.number_input("Maksimal Iterasi K-Means", value=50, step=5)

st.sidebar.divider()
st.sidebar.subheader("🔮 Parameter Peramalan")
hari_proyeksi = st.sidebar.slider("Durasi Proyeksi (Hari)", min_value=7, max_value=90, value=30, step=7)

# --- 5. PEMPROSESAN DATA ---
df_market, connection_status = fetch_real_market_data(ticker_symbol, period=time_period)

=======

#Pilihan Rentang Waktu
time_period = st.sidebar.selectbox("Rentang Waktu Histori", ["6m", "1y", "2y"], index=1)

st.sidebar.divider()
k_clusters = st.sidebar.slider("Jumlah Klaster (K)", min_value=2, max_value=5, value=3)
max_iter_input = st.sidebar.number_input("Maksimal Iterasi K-Means", value=50, step=5)

#4. PEMPROSESAN DATA 
df_market, connection_status = fetch_real_market_data(ticker_symbol, period=time_period)

#Tampilkan banner status koneksi API di atas dashboard
>>>>>>> 4bc216023ecf75d203eb237788c8c04b73b073cf
if "🟢" in connection_status:
    st.success(connection_status)
else:
    st.info(connection_status)

<<<<<<< HEAD
# ANTI KEYERROR: Ekstraksi fitur K-Means menggunakan nama kolom asli
X_features = df_market[['Daily_Return', 'Rolling_Vol']].values
=======
#Format nama kolom untuk kebutuhan grafik visualisasi
df_market.rename(columns={
    'Daily_Return': 'Keuntungan Harian (%)',
    'Rolling_Vol': 'Tingkat Volatilitas',
    'Cluster': 'Klaster'
}, inplace=True)

#Jalankan Klasterisasi K-Means murni
X_features = df_market[['Keuntungan Harian (%)', 'Tingkat Volatilitas']].values
>>>>>>> 4bc216023ecf75d203eb237788c8c04b73b073cf
labels, final_centroids = custom_kmeans(X_features, k=k_clusters, max_iters=max_iter_input)
df_market['Klaster'] = labels.astype(str)

<<<<<<< HEAD
# Ekstraksi Tren Masa Depan sebelum proses rename kolom dilakukan
df_tren, nilai_slope = hitung_tren_masa_depan(df_market, hari_ke_depan=hari_proyeksi)

# JURUS OUTER JOIN: Memisahkan komponen agar tidak terjadi konflik baris kosong (NaN)
df_h = pd.DataFrame({'Tanggal': df_market['Tanggal'], 'Harga Riil': df_market['Close']})
df_p = pd.DataFrame({'Tanggal': df_tren['Tanggal'], 'Proyeksi Tren': df_tren['Proyeksi Harga']})

# Siasat agar garis tren menyambung langsung dari titik historis terakhir harga riil
titik_jembatan = pd.DataFrame({
    'Tanggal': [df_market['Tanggal'].max()],
    'Proyeksi Tren': [df_market['Close'].values[-1]]
})
df_p = pd.concat([titik_jembatan, df_p], ignore_index=True).drop_duplicates(subset=['Tanggal'])

# Gabungkan data menggunakan teknik Outer Join dan urutkan berdasarkan Tanggal
df_visual_tren = pd.merge(df_h, df_p, on='Tanggal', how='outer').sort_values('Tanggal')

# Melakukan penamaan ulang kolom data historis untuk visualisasi chart klaster
df_market.rename(columns={
    'Daily_Return': 'Keuntungan Harian (%)',
    'Rolling_Vol': 'Tingkat Volatilitas',
    'Cluster': 'Klaster'
}, inplace=True)

# --- 6. VISUALISASI DASHBOARD DENGAN SISTEM TAB ---
tab1, tab2 = st.tabs(["🎯 Segmentasi Rezim Pasar", "🔮 Peramalan Tren Harga"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📉 Histori Pergerakan Harga")
        st.line_chart(data=df_market, x='Tanggal', y='Close', use_container_width=True)
        st.caption(f"Grafik harga penutupan historis untuk komponen {asset_choice}.")
    with col2:
        st.subheader("🎯 Pemetaan Rezim Pasar (Return vs Volatilitas)")
        st.scatter_chart(
            data=df_market,
            x='Keuntungan Harian (%)',
            y='Tingkat Volatilitas',
            color='Klaster',
            use_container_width=True
        )
        st.caption("Hasil pembagian rezim otomatis: Sumbu X (Persentase Return), Sumbu Y (Tingkat Gejolak/Volatilitas).")

with tab2:
    st.subheader(f"🔮 Proyeksi Arah Harga ke Depan ({hari_proyeksi} Hari)")
    st.line_chart(data=df_visual_tren, x='Tanggal', y=['Harga Riil', 'Proyeksi Tren'], use_container_width=True)
    
    # Menampilkan KPI Status Tren Finansial secara interaktif
    status_tren = "📈 MENAIK (Bullish)" if nilai_slope > 0 else "📉 MENURUN (Bearish)"
    c1, c2 = st.columns(2)
    c1.metric("Arah Tren Linear", status_tren)
    c2.metric("Kecepatan Perubahan (Slope)", f"{nilai_slope:.4f}")
    st.caption("Garis proyeksi dihitung murni menggunakan matriks kuadrat terkecil (Ordinary Least Squares) tanpa modul pihak ketiga.")

st.divider()
st.subheader("📋 Log Data Riwayat Pasar Terklaster")
st.dataframe(df_market[['Tanggal', 'Close', 'Keuntungan Harian (%)', 'Tingkat Volatilitas', 'Klaster']], use_container_width=True)
=======
#5. VISUALISASI DASHBOARD 
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"📉 Histori Pergerakan Harga")
    st.line_chart(data=df_market, x='Tanggal', y='Close', use_container_width=True)
    st.caption(f"Grafik harga penutupan historis untuk komponen {asset_choice}.")

with col2:
    st.subheader("🎯 Pemetaan Rezim Pasar (Return vs Volatilitas)")
    st.scatter_chart(
        data=df_market,
        x='Keuntungan Harian (%)',
        y='Tingkat Volatilitas',
        color='Klaster',
        use_container_width=True
    )
    st.caption("Hasil pembagian rezim otomatis: Sumbu X (Persentase Return), Sumbu Y (Tingkat Gejolak/Volatilitas).")

st.divider()
st.subheader("📋 Log Data Riwayat Pasar Terklaster")
st.dataframe(df_market[['Tanggal', 'Close', 'Keuntungan Harian (%)', 'Tingkat Volatilitas', 'Klaster']], use_container_width=True)
>>>>>>> 4bc216023ecf75d203eb237788c8c04b73b073cf
