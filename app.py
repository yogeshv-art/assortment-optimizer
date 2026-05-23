import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import io

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Assortment Optimizer",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Loose to Pack - Loose Pair Reduction")
st.caption("Production Planning + Inventory Utilization + SFG Visibility")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Upload Files")

mapping_file = st.sidebar.file_uploader(
    "Article_Color_Assort_Mapping.csv",
    type=["csv"]
)

assort_file = st.sidebar.file_uploader(
    "Assort_Code_Master.csv",
    type=["csv"]
)

stock_file = st.sidebar.file_uploader(
    "Current_Pairs_Stock.csv",
    type=["csv"]
)

sfg_file = st.sidebar.file_uploader(
    "SFG_Stock.csv",
    type=["csv"]
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_gap(gap_dict):

    if not gap_dict:
        return "—"

    return ", ".join(
        f"Size {s}: +{q}"
        for s, q in sorted(gap_dict.items())
    )

# ============================================================
# MAIN ENGINE
# ============================================================

if mapping_file and assort_file and stock_file and sfg_file:

    try:

        with st.spinner("Running Optimization Engine..."):

            # ====================================================
            # LOAD FILES
            # ====================================================

            mapping = pd.read_csv(mapping_file)
            assort_master = pd.read_csv(assort_file)
            stock = pd.read_csv(stock_file)
            sfg_stock = pd.read_csv(sfg_file)

            # ====================================================
            # VALIDATION
            # ====================================================

            required_mapping = ['Article', 'Colour', 'Assortment']
            required_assort = ['ASST CODE', 'Size', 'Qty']
            required_stock = ['Article Code', 'Colour Code', 'Size', 'Qty', 'Location']

            required_sfg = [
                'Article Code',
                'Assortment Size',
                'Unrestricted'
            ]

            missing_mapping = [
                c for c in required_mapping
                if c not in mapping.columns
            ]

            missing_assort = [
                c for c in required_assort
                if c not in assort_master.columns
            ]

            missing_stock = [
                c for c in required_stock
                if c not in stock.columns
            ]

            missing_sfg = [
                c for c in required_sfg
                if c not in sfg_stock.columns
            ]

            if missing_mapping:
                st.error(f"Missing Mapping Columns: {missing_mapping}")
                st.stop()

            if missing_assort:
                st.error(f"Missing Assort Columns: {missing_assort}")
                st.stop()

            if missing_stock:
                st.error(f"Missing Stock Columns: {missing_stock}")
                st.stop()

            if missing_sfg:
                st.error(f"Missing SFG Columns: {missing_sfg}")
                st.stop()

            # ====================================================
            # PREPROCESSING
            # ====================================================

            assort_master['Qty'] = assort_master['Qty'].fillna(0).astype(int)
            assort_master['Size'] = assort_master['Size'].astype(int)

            stock['Size'] = stock['Size'].astype(int)

            sfg_stock = sfg_stock.rename(columns={
                'Article Code': 'SFG Article',
                'Assortment Size': 'Size',
                'Unrestricted': 'SFG Stock'
            })

            sfg_stock['Size'] = pd.to_numeric(
                sfg_stock['Size'],
                errors='coerce'
            ).fillna(0).astype(int)

            sfg_stock['SFG Stock'] = pd.to_numeric(
                sfg_stock['SFG Stock'],
                errors='coerce'
            ).fillna(0).astype(int)

            sfg_stock = sfg_stock[
                sfg_stock['SFG Stock'] > 0
            ]

            mapping['Art_Col'] = (
                mapping['Article'].astype(str)
                + "_"
                + mapping['Colour'].astype(str)
            )

            stock['Art_Col'] = (
                stock['Article Code'].astype(str)
                + "_"
                + stock['Colour Code'].astype(str)
            )

            valid_keys = set(mapping['Art_Col'].unique())

            stock = stock[
                stock['Art_Col'].isin(valid_keys)
            ]

            asst_lookup = {
                code: grp.set_index('Size')['Qty'].to_dict()
                for code, grp in assort_master.groupby('ASST CODE')
            }

            # ====================================================
            # DASHBOARD METRICS
            # ====================================================

            total_articles = mapping['Art_Col'].nunique()
            total_stock = stock['Qty'].sum()
            total_locations = stock['Location'].nunique()

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Article-Colour", total_articles)
            col2.metric("Total Pairs Stock", f"{int(total_stock):,}")
            col3.metric("Total Locations", total_locations)

            # ====================================================
            # CORE ENGINE
            # ====================================================

            results = []

            groups = list(mapping.groupby('Art_Col'))

            progress_bar = st.progress(0)

            for i, (art_col, group) in enumerate(groups):

                progress_bar.progress((i + 1) / len(groups))

                valid_assts = [
                    a for a in group['Assortment'].unique()
                    if a in asst_lookup
                ]

                if not valid_assts:
                    continue

                group_stock = stock[
                    stock['Art_Col'] == art_col
                ]

                stock_map = group_stock.groupby('Size')['Qty'].sum().to_dict()

                all_sizes = sorted(
                    list(
                        set(stock_map.keys()) |
                        {
                            s
                            for a in valid_assts
                            for s in asst_lookup[a].keys()
                        }
                    )
                )

                size_idx = {
                    s: i
                    for i, s in enumerate(all_sizes)
                }

                # ====================================================
                # MILP OPTIMIZATION
                # ====================================================

                n_vars = len(valid_assts)

                c = -np.ones(n_vars)

                A = np.zeros((len(all_sizes), n_vars))

                b_u = np.array([
                    stock_map.get(s, 0)
                    for s in all_sizes
                ])

                for j, asst in enumerate(valid_assts):

                    for size, qty in asst_lookup[asst].items():

                        A[size_idx[size], j] = qty

                res = milp(
                    c=c,
                    constraints=LinearConstraint(A, 0, b_u),
                    integrality=np.ones(n_vars),
                    bounds=Bounds(0, np.inf)
                )

                current_packs = (
                    res.x if res.success
                    else np.zeros(n_vars)
                )

                used_stock = A @ current_packs

                optimized_pool = b_u - used_stock

                # ====================================================
                # RESULT GENERATION
                # ====================================================

                for j, asst in enumerate(valid_assts):

                    p_size = sum(asst_lookup[asst].values())

                    asst_need = asst_lookup[asst]

                    gap = {}

                    for size, req in asst_need.items():

                        avail = optimized_pool[size_idx[size]]

                        if avail < req:
                            gap[size] = int(req - avail)

                    produce_str = format_gap(gap)

                    # ================================================
                    # SFG STOCK CHECK
                    # ================================================

                    sfg_article_list = []
                    sfg_stock_list = []

                    if produce_str != '—':

                        for item in produce_str.split(","):

                            item = item.strip()

                            size_part, qty_part = item.split(":")

                            size = int(
                                size_part.replace("Size", "").strip()
                            )

                            matched_rows = sfg_stock[
                                sfg_stock['Size'] == size
                            ]

                            if not matched_rows.empty:

                                for _, row in matched_rows.iterrows():

                                    sfg_article_list.append(
                                        str(row['SFG Article'])
                                    )

                                    sfg_stock_list.append(
                                        str(row['SFG Stock'])
                                    )

                    sfg_articles = (
                        ", ".join(sfg_article_list)
                        if sfg_article_list else "—"
                    )

                    sfg_stocks = (
                        ", ".join(sfg_stock_list)
                        if sfg_stock_list else "—"
                    )

                    # ================================================
                    # APPEND RESULTS
                    # ================================================

                    results.append({
                        'Article': art_col.split('_')[0],
                        'Colour': art_col.split('_')[1],
                        'Assortment': asst,
                        'Pack Size': p_size,
                        'Current Packs': int(current_packs[j]),
                        'Production Needed': produce_str,
                        'SFG Articles': sfg_articles,
                        'SFG Stock Available': sfg_stocks
                    })

            # ====================================================
            # OUTPUT
            # ====================================================

            final_df = pd.DataFrame(results)

            st.success("Optimization Completed Successfully")

            st.subheader("Optimization Results")

            st.dataframe(
                final_df,
                use_container_width=True,
                height=650
            )

            # ====================================================
            # DOWNLOAD BUTTON
            # ====================================================

            csv = final_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Optimization CSV",
                data=csv,
                file_name="Scenario_Assortment_Plan.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error("Error Occurred")

        st.exception(e)

else:

    st.info("Please upload all 4 CSV files to begin analysis.")
