import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(page_title="EDA Explorer", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis (EDA) App")
st.markdown("Upload a **CSV** or **Excel** file to begin exploring your data.")

# ─────────────────────────────────────────
# Task 1 – Accept CSV or Excel
# ─────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: .csv, .xlsx, .xls"
)

@st.cache_data
def load_data(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format.")
        return None

# ─────────────────────────────────────────
# Main app
# ─────────────────────────────────────────
if uploaded_file is not None:
    data = load_data(uploaded_file)

    if data is not None:

        # ── Overview ──────────────────────────────────────────────────
        st.header("🔍 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows",    data.shape[0])
        col2.metric("Columns", data.shape[1])
        col3.metric("Missing Values", int(data.isnull().sum().sum()))

        st.subheader("First 5 Rows")
        st.dataframe(data.head())

        st.subheader("Data Types")
        st.dataframe(data.dtypes.rename("dtype").reset_index().rename(columns={"index": "Column"}))

        # ── Numerical Summary ─────────────────────────────────────────
        st.header("📐 Numerical Feature Summary")
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            st.dataframe(data[num_cols].describe())
        else:
            st.info("No numerical features found in this dataset.")

        # ── Task 2 – Non-numerical Summary (only if they exist) ────────
        cat_cols = data.select_dtypes(include=["bool", "object"]).columns.tolist()
        if cat_cols:
            st.header("🔤 Non-Numerical Feature Summary")
            st.dataframe(data[cat_cols].describe())
        # If no categorical columns → section is silently skipped (no error)

        # ── Missing Values ─────────────────────────────────────────────
        st.header("❓ Missing Values")
        missing = data.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            miss_df = missing.reset_index()
            miss_df.columns = ["Column", "Missing Count"]
            miss_df["Missing %"] = (miss_df["Missing Count"] / len(data) * 100).round(2)
            st.dataframe(miss_df)

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=miss_df, x="Column", y="Missing Count", ax=ax, palette="Reds_r")
            ax.set_title("Missing Values per Column")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
            plt.close()
        else:
            st.success("✅ No missing values found!")

        # ── Task 3 – Extra Graphs ──────────────────────────────────────
        st.header("📈 Visualizations")

        if num_cols:
            # 1. Histograms for all numerical columns
            st.subheader("1. Histograms (Numerical Features)")
            n = len(num_cols)
            cols_per_row = 3
            rows = (n + cols_per_row - 1) // cols_per_row
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))
            axes = np.array(axes).flatten()
            for i, col in enumerate(num_cols):
                axes[i].hist(data[col].dropna(), bins=30, color="#4C72B0", edgecolor="white")
                axes[i].set_title(col, fontsize=10)
                axes[i].set_xlabel("")
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            fig.suptitle("Distribution of Numerical Features", fontsize=14, y=1.01)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # 2. Box plots
            st.subheader("2. Box Plots (Outlier Detection)")
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))
            axes = np.array(axes).flatten()
            for i, col in enumerate(num_cols):
                axes[i].boxplot(data[col].dropna(), patch_artist=True,
                                boxprops=dict(facecolor="#4C72B0", color="white"),
                                medianprops=dict(color="yellow", linewidth=2))
                axes[i].set_title(col, fontsize=10)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            fig.suptitle("Box Plots of Numerical Features", fontsize=14, y=1.01)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # 3. Correlation Heatmap
            if len(num_cols) >= 2:
                st.subheader("3. Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(max(8, len(num_cols)), max(6, len(num_cols) - 1)))
                corr = data[num_cols].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                            ax=ax, linewidths=0.5, vmin=-1, vmax=1)
                ax.set_title("Feature Correlation Matrix")
                st.pyplot(fig)
                plt.close()

            # 4. Scatter Plot (user-selectable axes)
            if len(num_cols) >= 2:
                st.subheader("4. Scatter Plot")
                sc1, sc2 = st.columns(2)
                x_axis = sc1.selectbox("X-axis", num_cols, index=0)
                y_axis = sc2.selectbox("Y-axis", num_cols, index=min(1, len(num_cols) - 1))
                hue_col = None
                if cat_cols:
                    hue_col = st.selectbox("Color by (optional)", ["None"] + cat_cols)
                    hue_col = None if hue_col == "None" else hue_col

                fig, ax = plt.subplots(figsize=(8, 5))
                if hue_col:
                    for label, grp in data.groupby(hue_col):
                        ax.scatter(grp[x_axis], grp[y_axis], label=label, alpha=0.7, s=40)
                    ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc="upper left")
                else:
                    ax.scatter(data[x_axis], data[y_axis], alpha=0.7, s=40, color="#4C72B0")
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"{x_axis} vs {y_axis}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # 5. Pair Plot (up to 5 columns for performance)
            if len(num_cols) >= 2:
                st.subheader("5. Pair Plot (up to 5 numerical features)")
                pair_cols = num_cols[:5]
                fig = sns.pairplot(data[pair_cols].dropna(), diag_kind="kde", plot_kws={"alpha": 0.5})
                fig.fig.suptitle("Pair Plot", y=1.02)
                st.pyplot(fig)
                plt.close()

        # 6. Bar chart for categorical columns
        if cat_cols:
            st.subheader("6. Category Value Counts")
            selected_cat = st.selectbox("Select a categorical column", cat_cols)
            vc = data[selected_cat].value_counts().head(20)
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, palette="viridis")
            ax.set_title(f"Top categories in '{selected_cat}'")
            ax.set_xlabel(selected_cat)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Download cleaned summary ──────────────────────────────────
        st.header("⬇️ Download Summary")
        if num_cols:
            csv_summary = data[num_cols].describe().to_csv().encode("utf-8")
            st.download_button("Download Numerical Summary (CSV)",
                               data=csv_summary,
                               file_name="numerical_summary.csv",
                               mime="text/csv")

else:
    st.info("👆 Please upload a CSV or Excel file to get started.")