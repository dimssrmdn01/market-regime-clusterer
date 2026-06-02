import streamlit as st
import pandas as pd
import numpy as np

#PAGE CONFIGURATION
st.set_page_config(page_title="Market Volatility Clusterer", layout="wide")
st.title("📈 Market Volatility Clustering Engine (Pure NumPy K-Means)")
st.markdown("An unsupervised machine learning dashboard to cluster market regimes based on returns and volatility.")
st.divider()

#1. SIMULASI DATA PASAR 
@st.cache_data
def generate_market_data():
    """Generates synthetic crypto/forex market data with distinct regimes."""
    np.random.seed(42)
    n_days = 200
    
    #Kumpulan kondisi pasar (Sideways, Trending, High Volatility)
    regime_1 = np.random.normal(loc=0.0, scale=0.5, size=80)    
    regime_2 = np.random.normal(loc=0.2, scale=1.2, size=60)    
    regime_3 = np.random.normal(loc=-0.4, scale=1.8, size=60)   
    
    all_returns = np.concatenate([regime_1, regime_2, regime_3])
    
    #Rekonstruksi menjadi pergerakan harga akumulatif
    price = 1000
    price_history = []
    for r in all_returns:
        price += price * (r / 100)
        price_history.append(price)
        
    df = pd.DataFrame({
        'Day': np.arange(1, n_days + 1),
        'Price': price_history,
        'Daily_Return': all_returns,
        'Rolling_Vol': pd.Series(all_returns).rolling(window=5, min_periods=1).std().fillna(0).values
    })
    return df

df_market = generate_market_data()

#2. ALGORITMA K-MEANS FROM SCRATCH 
def custom_kmeans(X, k, max_iters=100):
    """Pure NumPy implementation of K-Means Clustering."""
    n_samples = X.shape[0]
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

#3. SIDEBAR: KONTROL PARAMETER 
st.sidebar.header("🎯 Cluster Parameters")
k_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=5, value=3)
max_iter_input = st.sidebar.number_input("Max K-Means Iterations", value=50, step=5)

#4. DATA PREPARATION & PROCESSING
X_features = df_market[['Daily_Return', 'Rolling_Vol']].values
labels, final_centroids = custom_kmeans(X_features, k=k_clusters, max_iters=max_iter_input)
df_market['Cluster'] = labels.astype(str)

#5. DASHBOARD DISPLAY 
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📉 Market Price History")
    st.line_chart(df_market.set_index("Day")["Price"], use_container_width=True)

with col2:
    st.subheader("🎯 Feature Space Clustering (Return vs Volatility)")
    st.scatter_chart(
        data=df_market,
        x='Daily_Return',
        y='Rolling_Vol',
        color='Cluster',
        use_container_width=True
    )

st.divider()
st.subheader("📋 Real-Time Clustered Market Logs")
st.dataframe(df_market, use_container_width=True)