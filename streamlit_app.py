import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import plotly.express as px


# ======================
# UI BLOCK
# ======================
st.title("🛍️ Customer Segmentation Dashboard")
st.markdown("Interactive clustering with KMeans + dimensionality reduction (PCA / UMAP)")


# ======================
# CONTROLS (SIDEBAR)
# ======================
clusters_num = st.sidebar.slider("Clusters (KMeans)", min_value=2, max_value=10, value=10)
reduce_type = st.sidebar.selectbox("Choose projection method", ["PCA", "UMAP"])


# ======================
# DATA LOADING
# ======================
@st.cache_data
def get_data():
    data = pd.read_csv("Mall_Customers.csv")

    # drop useless id
    data = data.drop(columns=["CustomerID"])

    # encode gender manually
    data["Gender"] = data["Gender"].apply(lambda x: 1 if x == "Female" else 0)

    features = data.iloc[:, :]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    return data, scaled


raw_df, scaled_x = get_data()


# ======================
# CLUSTERING
# ======================
model_km = KMeans(n_clusters=clusters_num, random_state=42)
cluster_ids = model_km.fit_predict(scaled_x)


# ======================
# DIMENSION REDUCTION
# ======================
if reduce_type == "PCA":
    reducer = PCA(n_components=2)
    proj = reducer.fit_transform(scaled_x)
    axis_a, axis_b = "PC-A", "PC-B"
else:
    reducer = umap.UMAP(n_components=2, random_state=42)
    proj = reducer.fit_transform(scaled_x)
    axis_a, axis_b = "Dim-1", "Dim-2"


# ======================
# BUILD PLOT DATA
# ======================
plot_frame = pd.DataFrame(proj, columns=[axis_a, axis_b])
plot_frame["group"] = cluster_ids.astype(str)


# ======================
# VISUALIZATION
# ======================
fig = px.scatter(
    plot_frame,
    x=axis_a,
    y=axis_b,
    color="group",
    title=f"Customer Groups ({reduce_type}) | K={clusters_num}",
    opacity=0.8
)

st.plotly_chart(fig, use_container_width=True)



# ======================
# COMPARISON VIEW (PCA vs UMAP)
# ======================

col1, col2 = st.columns(2)

# -------- PCA --------
with col1:
    pca_model = PCA(n_components=2)
    pca_proj = pca_model.fit_transform(scaled_x)

    pca_df = pd.DataFrame(pca_proj, columns=["PC1", "PC2"])
    pca_df["cluster"] = cluster_ids.astype(str)

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="cluster",
        title="PCA View",
        opacity=0.8
    )

    st.plotly_chart(fig_pca, use_container_width=True)


# -------- UMAP --------
with col2:
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_proj = umap_model.fit_transform(scaled_x)

    umap_df = pd.DataFrame(umap_proj, columns=["UMAP1", "UMAP2"])
    umap_df["cluster"] = cluster_ids.astype(str)

    fig_umap = px.scatter(
        umap_df,
        x="UMAP1",
        y="UMAP2",
        color="cluster",
        title="UMAP View",
        opacity=0.8
    )

    st.plotly_chart(fig_umap, use_container_width=True)
