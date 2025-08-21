
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("🛍 Customer Segmentation Dashboard")
st.caption("Upload Mall_Customers.csv (or keep it next to this app) and pick k to see segments.")

# -----------------------------
# 1) Load data
# -----------------------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload CSV (Mall_Customers.csv)", type=["csv"])
    k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=5, step=1)
    show_eda = st.checkbox("Show EDA (histograms & pairplot)", value=True)
    show_model_diag = st.checkbox("Show Model Diagnostics (Elbow & Silhouette)", value=True)

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    # Fallback: try local file
    return pd.read_csv("Mall_Customers.csv")

try:
    data = load_data(uploaded)
except Exception as e:
    st.error("Couldn't find or read a CSV. Please upload 'Mall_Customers.csv' using the sidebar.")
    st.stop()

# Basic sanity
required_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}. It must include {required_cols}.")
    st.stop()

st.write("### Dataset Preview")
st.dataframe(data.head())

# Optionally encode Gender (not used in clustering)
if "Gender" in data.columns:
    data["Gender_num"] = data["Gender"].map({"Male": 0, "Female": 1})

# -----------------------------
# 2) EDA
# -----------------------------
if show_eda:
    st.subheader("Feature Distributions")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots()
        sns.histplot(data["Age"].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots()
        sns.histplot(data["Annual Income (k$)"].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title("Annual Income (k$)")
        st.pyplot(fig)

    with c3:
        fig, ax = plt.subplots()
        sns.histplot(data["Spending Score (1-100)"].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title("Spending Score (1-100)")
        st.pyplot(fig)

    with st.expander("Pairplot (Age, Income, Spending Score)"):
        g = sns.pairplot(
            data[required_cols].dropna(),
            diag_kind="kde",
            plot_kws={"alpha": 0.7, "s": 40}
        )
        st.pyplot(g.fig)

# -----------------------------
# 3) Clustering
# -----------------------------
X = data[required_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_scaled)

data_clustered = data.copy()
data_clustered["Cluster"] = labels

# PCA for 2D viz
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

st.subheader("Customer Segments (PCA view)")
fig, ax = plt.subplots()
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data_clustered["Cluster"], palette="Set1", ax=ax)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Clusters in PCA Space")
st.pyplot(fig)

# Cluster profiles
st.subheader("Cluster Profiles (Averages)")
profiles = data_clustered.groupby("Cluster")[required_cols].mean().round(2)
st.dataframe(profiles)

# Counts
st.write("#### Customers per Cluster")
counts = data_clustered["Cluster"].value_counts().sort_index()
st.bar_chart(counts)

# -----------------------------
# 4) Diagnostics
# -----------------------------
if show_model_diag:
    with st.expander("Elbow Method (SSE)"):
        sse = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, random_state=42, n_init="auto")
            km.fit(X_scaled)
            sse.append(km.inertia_)
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), sse, marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("SSE")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

    with st.expander("Silhouette Scores"):
        sil_scores = []
        ks = list(range(2, 11))
        for i in ks:
            km = KMeans(n_clusters=i, random_state=42, n_init="auto")
            lbls = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, lbls)
            sil_scores.append(sil)
        fig, ax = plt.subplots()
        ax.plot(ks, sil_scores, marker="o")
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Scores vs k")
        st.pyplot(fig)

# -----------------------------
# 5) Download
# -----------------------------
csv_bytes = data_clustered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Clustered CSV",
    data=csv_bytes,
    file_name="Mall_Customers_with_Clusters.csv",
    mime="text/csv"
)

st.success("Done! Tip: try different k from the sidebar and compare the profiles.")
