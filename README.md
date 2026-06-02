#  Market Volatility Clustering Engine (Pure NumPy K-Means)

An interactive, web-based unsupervised machine learning dashboard designed to analyze and segment financial market regimes (Crypto/Forex) based on daily returns and rolling volatility. 

Instead of relying on standard machine learning libraries like `scikit-learn`, the **K-Means Clustering algorithm in this project was built entirely from scratch using pure NumPy**, demonstrating a deep foundational understanding of vector calculations, distance metrics, and iterative optimization.

## 🚀 Live Demo
You can access the live interactive dashboard here: **[https://dimss-market-regime.streamlit.app/]**

##  Key Features
- **Algorithm From Scratch:** Complete Python implementation of the K-Means clustering algorithm using vectorization in NumPy.
- **Dynamic Market Simulator:** Generates synthetic asset price data modeling 3 distinct realistic market regimes: *Sideways/Low Volatility*, *Bull Run/High Volatility*, and *Panic Selling/Crash Zone*.
- **Interactive UI (Streamlit):** Real-time cluster configuration where users can adjust the number of clusters ($K$) and maximum iterations seamlessly via the sidebar.
- **Robust Data Handling:** Built-in safeguards against statistical mathematical edge cases (such as zero division and `NaN` values in rolling computations).

##  Mathematical Approach
The custom engine utilizes the **Euclidean Distance** to assign data points to their nearest centroids:

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

Centroids are iteratively updated by calculating the mean of all assigned vector coordinates until the cluster centroids converge completely ($\Delta C \approx 0$).

##  Tech Stack & Requirements
- **Python 3.11+**
- **Streamlit** (Dashboard UI & Interactive Charts)
- **NumPy** (Mathematical Vectorized Engine)
- **Pandas** (Data Wrangling & Rolling Statistics)

##  Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/market-regime-clusterer.git](https://github.com/YOUR_USERNAME/market-regime-clusterer.git)
   cd market-regime-clusterer
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run the Application:**
   ```bash
   streamlit run market_clusterer.py

Developed for academic research and data portfolio enhancement in Data Science & Applied Statistics.
