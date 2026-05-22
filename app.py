import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Assortment Optimizer",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Loose to Pack - Loose Pair Reduction")
st.caption("Production Planning + Inventory Utilization + Assortment Recommendation")

# ============================================================
# SIDEBAR FILE UPLOADS
# ============================================================

st.sidebar.header("Upload Files")

mapping_file = st.sidebar.file_uploader(
    "Upload Article_Color_Assort_Mapping.csv",
    type=["csv"]
)

assort_file = st.sidebar.file_uploader(
    "Upload Assort_Code_Master.csv",
    type=["csv"]
)

stock_file = st.sidebar.file_uploader(
    "Upload Current_Pairs_Stock.csv",
    type=["csv"]
)

fg_stock_file = st.sidebar.file_uploader(
    "Upload FG Stock File (Optional)",
    type=["csv"]
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allocate_from_locations(size_need: dict, loc_stock_df: pd.DataFrame):

    pool = {}

    for _, row in loc_stock_df.iterrows():
        key = (row['Location'], row['Size'])
        pool[key] = pool.get(key, 0) + int(row['Qty'])

    remaining_need = {s: q for s, q in size_need.items() if q > 0}

    allocation = {}

    for (loc, size), avail in sorted(pool.items()):

        if size not in remaining_need:
            continue

        needed = remaining_need[size]

        take = min(avail, needed)

        if take > 0:

            allocation[(loc, size)] = take

            remaining_need[size] -= take

            if remaining_need[size] == 0:
                del remaining_need[size]

    loc_groups = {}

    for (loc, size), qty in sorted(allocation.items()):
        loc_groups.setdefault(loc, []).append(f"Size {size}={qty}")

    allocation_str = " | ".join(
        f"{loc}: {', '.join(parts)}"
        for loc, parts in sorted(loc_groups.items())
    ) if loc_groups else "—"

    return allocation_str, remaining_need


def format_gap(gap_dict: dict) -> str:

    if not gap_dict:
        return "—"

    return ", ".join(
        f"Size {s}: +{q}"
        for s, q in sorted(gap_dict.items())
    )

# ============================================================
# MAIN ENGINE
# ============================================================

if mapping_file and assort_file and stock_file:

    try:

        with st.spinner("Running Optimization Engine..."):

            # ====================================================
            # LOAD FILES
            # ====================================================

            mapping = pd.read_csv(mapping_file)
            assort_master = pd.read_csv(assort_file)
            stock = pd.read_csv(stock_file)

            # ====================================================
            # FG STOCK LOAD
            # ====================================================

            fg_stock = None
            fg_lookup = {}

            if fg_stock_file is not None:

                fg_stock = pd.read_csv(fg_stock_file)

                required_fg = [
                    'Article Code',
                    'Assortment Size',
                    'Unrestricted'
                ]

                missing_fg = [
                    c for c in required_fg
                    if c not in fg_stock.columns
                ]

                if missing_fg:
                    st.error(f"Missing FG Stock Columns: {missing_fg}")
                    st.stop()

                fg_stock = fg_stock.rename(columns={
                    'Article Code': 'Article',
                    'Assortment Size': 'Size',
                    'Unrestricted': 'Qty'
                })

                fg_stock['Size'] = pd.to_numeric(
                    fg_stock['Size'],
                    errors='coerce'
                ).fillna(0).astype(int)

                fg_stock['Qty'] = pd.to_numeric(
                    fg_stock['Qty'],
                    errors='coerce'
                ).fillna(0).astype(int)

                fg_stock = fg_stock[
                    fg_stock['Qty'] > 0
                ]

                for (article, size), grp in fg_stock.groupby(
                    ['Article', 'Size']
                ):

                    fg_lookup[
                        (str(article), int(size))
                    ] = grp['Qty'].sum()

            # ====================================================
            # VALIDATION
            # ====================================================

            required_mapping = [
                'Article',
                'Colour',
                'Assortment'
            ]

            required_assort = [
                'ASST CODE',
                'Size',
                'Qty'
            ]

            required_stock = [
                'Article Code',
                'Colour Code',
                'Size',
                'Qty',
                'Location'
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

            if missing_mapping:
                st.error(f"Missing Mapping Columns: {missing_mapping}")
                st.stop()

            if missing_assort:
                st.error(f"Missing Assort Columns: {missing_assort}")
                st.stop()

            if missing_stock:
                st.error(f"Missing Stock Columns: {missing_stock}")
                st.stop()

            # ====================================================
            # PREPROCESSING
            # ====================================================

            assort_master['Qty'] = assort_master['Qty'].fillna(0).astype(int)

            assort_master['Size'] = assort_master['Size'].astype(int)

            stock['Size'] = stock['Size'].astype(int)

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
            # KPI DASHBOARD
            # ====================================================

            total_articles = mapping['Art_Col'].nunique()
            total_stock = stock['Qty'].sum()
            total_locations = stock['Location'].nunique()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Article-Colour",
                total_articles
            )

            col2.metric(
                "Total Pairs Stock",
                f"{int(total_stock):,}"
            )

            col3.metric(
                "Total Locations",
                total_locations
            )

            # ====================================================
            # CORE ENGINE
            # ====================================================

            results = []

            groups = list(mapping.groupby('Art_Col'))

            progress_bar = st.progress(0)

            for i, (art_col, group) in enumerate(groups):

                progress_bar.progress(
                    (i + 1) / len(groups)
                )

                valid_assts = [
                    a for a in group['Assortment'].unique()
                    if a in asst_lookup
                ]

                if not valid_assts:
                    continue

                group_stock = stock[
                    stock['Art_Col'] == art_col
                ]

                stock_map = group_stock.groupby(
                    'Size'
                )['Qty'].sum().to_dict()

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
                # MILP
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

                    p_size = sum(
                        asst_lookup[asst].values()
                    )

                    asst_need = asst_lookup[asst]

                    gap = {}

                    for size, req in asst_need.items():

                        avail = optimized_pool[
                            size_idx[size]
                        ]

                        if avail < req:
                            gap[size] = int(req - avail)

                    produce_str = format_gap(gap)

                    # ================================================
                    # FG STOCK CHECK
                    # ================================================

                    fg_status = "No FG Data"
                    fg_adjusted = produce_str

                    if (
                        fg_stock is not None
                        and produce_str != '—'
                    ):

                        article = str(
                            art_col.split('_')[0]
                        )

                        needed = {}

                        for item in produce_str.split(","):

                            item = item.strip()

                            size_part, qty_part = item.split(":")

                            size = int(
                                size_part.replace(
                                    "Size", ""
                                ).strip()
                            )

                            qty = int(
                                qty_part.replace(
                                    "+", ""
                                ).strip()
                            )

                            needed[size] = qty

                        remaining_fg = {}

                        for size, qty in needed.items():

                            available = fg_lookup.get(
                                (article, size),
                                0
                            )

                            remaining_fg[size] = max(
                                0,
                                qty - available
                            )

                        if all(
                            v == 0
                            for v in remaining_fg.values()
                        ):

                            fg_status = "FG Covers Fully"
                            fg_adjusted = "—"

                        else:

                            fg_status = "Partial FG Available"

                            fg_adjusted = ", ".join(
                                f"Size {s}: +{q}"
                                for s, q in remaining_fg.items()
                                if q > 0
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
                        'Produce (Size:Qty)': produce_str,
                        'FG Status': fg_status,
                        'Adjusted Production (After FG)': fg_adjusted
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
                height=600
            )

            # ====================================================
            # DOWNLOAD
            # ====================================================

            csv = final_df.to_csv(
                index=False
            ).encode('utf-8')

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

    st.info(
        "Please upload all required CSV files to begin analysis."
    )