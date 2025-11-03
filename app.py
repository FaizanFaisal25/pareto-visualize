import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Pareto Data Duplication', layout='wide')
sns.set_theme(style='whitegrid')

st.title('Dataset Duplication with Pareto Distribution')

# --- Sidebar Inputs ---
st.sidebar.header('Dataset Settings')
N_total = st.sidebar.slider('Total number of unique points', min_value=500, max_value=10000, value=5000, step=100)
split_ratio = st.sidebar.slider('Train/Test split ratio', min_value=0.1, max_value=0.9, value=0.8, step=0.05)

st.sidebar.header('Pareto Distribution Parameters')
gamma = st.sidebar.slider('Pareto shape parameter (gamma)', min_value=0.5, max_value=5.0, value=2.0, step=0.1)
x_m = st.sidebar.slider('Pareto scale parameter (x_m)', min_value=1, max_value=10, value=1, step=1)

# --- Create Dummy Dataset of unique points ---
df = pd.DataFrame({'PointID': range(1, N_total+1)})

# --- Train/Test Split ---
n_train = int(len(df) * split_ratio)
n_test = len(df) - n_train
train_df = df.iloc[:n_train].copy()
test_df = df.iloc[n_train:].copy()

st.write(f'Total unique points: {len(df)}')
st.write(f'Train unique points: {len(train_df)}')
st.write(f'Test unique points: {len(test_df)}')

# --- Duplication via Pareto applied to train set ---
np.random.seed(42)
train_counts = (np.random.pareto(a=gamma, size=len(train_df)) + 1) * x_m
train_counts = train_counts.astype(int)

# --- Build duplicated train dataset ---
duplicated_train = pd.DataFrame(np.repeat(train_df.values, train_counts, axis=0), columns=train_df.columns)

st.write(f'Total duplicated train size: {len(duplicated_train)}')

# --- Compute Monofact Rate ---
monofact_rate = np.mean(train_counts == 1) * 100
st.write(f'Monofact rate (percent of items appearing only once in training set): {monofact_rate:.2f}%')

# --- Visualization: Line plot with circle markers ---
st.subheader('Number of Appearances vs Frequency')
count_freq = pd.Series(train_counts).value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8,3.5))
ax.plot(count_freq.index, count_freq.values, marker='o', linestyle='-', color='teal')
ax.set_xlabel('Number of appearances (duplications)')
ax.set_ylabel('Frequency (number of points)')
ax.set_yscale('log')
ax.set_title('Number of Appearances vs Frequency for Train Points')

# --- Center the graph ---
col1, col2, col3 = st.columns([1, 4, 1])  # middle column is wider
with col2:
    st.pyplot(fig)
