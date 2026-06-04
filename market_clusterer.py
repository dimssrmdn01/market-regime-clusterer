import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

#KONFIGURASI HALAMAN 
st.set_page_config(page_title="Analisis Volatilitas Pasar", layout="wide")
st.title("📈 Mesin Klasterisasi Volatilitas Pasar (K-Means Murni)")
st.markdown("Dashboard *machine learning* tanpa pengawasan (*unsupervised*) untuk mengelompokkan rezim pasar berdasarkan tingkat keuntungan dan volatilitas.")
st.divider()

#1. UNDUH DATA PASAR DENGAN SISTEM CADANGAN (AUTOMATIC FALLBACK)
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
        
        #JIKA BERHASIL TERHUBUNG KE YAHOO FINANCE
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
        
    #RENCANA CADANGAN: JIKA YAHOO MEMBLOKIR (SIMULASI STOKASTIK REPRISAL) 
    np.random.seed(42)
    n_days = 250 if period == "1y" else (125 if period == "6m" else 500)
    
    #Karakteristik volatilitas & harga awal tiap aset
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
        
    #Pembuatan data tiruan menggunakan Geometric Brownian Motion
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

#2. ALGORITMA K-MEANS BUATAN SENDIRI (FROM SCRATCH) 
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

#3. SIDEBAR: KONTROL PARAMETER & PILIHAN ASET 
st.sidebar.header("🎛️ Kontrol Pasar & Model")

#Menu Pilihan Aset
asset_mapping = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Emas (GC=F)": "GC=F",
    "Saham Apple (AAPL)": "AAPL",
    "Indeks S&P 500 (^GSPC)": "^GSPC"
}
asset_choice = st.sidebar.selectbox("Pilih Aset Pasar", list(asset_mapping.keys()))
ticker_symbol = asset_mapping[asset_choice]

#Pilihan Rentang Waktu
time_period = st.sidebar.selectbox("Rentang Waktu Histori", ["6m", "1y", "2y"], index=1)

st.sidebar.divider()
k_clusters = st.sidebar.slider("Jumlah Klaster (K)", min_value=2, max_value=5, value=3)
max_iter_input = st.sidebar.number_input("Maksimal Iterasi K-Means", value=50, step=5)

#4. PEMPROSESAN DATA 
df_market, connection_status = fetch_real_market_data(ticker_symbol, period=time_period)

#Tampilkan banner status koneksi API di atas dashboard
if "🟢" in connection_status:
    st.success(connection_status)
else:
    st.info(connection_status)

#Format nama kolom untuk kebutuhan grafik visualisasi
df_market.rename(columns={
    'Daily_Return': 'Keuntungan Harian (%)',
    'Rolling_Vol': 'Tingkat Volatilitas',
    'Cluster': 'Klaster'
}, inplace=True)

#Jalankan Klasterisasi K-Means murni
X_features = df_market[['Keuntungan Harian (%)', 'Tingkat Volatilitas']].values
labels, final_centroids = custom_kmeans(X_features, k=k_clusters, max_iters=max_iter_input)
df_market['Klaster'] = labels.astype(str)

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
