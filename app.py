import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import io

st.set_page_config(
    page_title="Assortment Optimizer",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Loose to Pack  - Loose Pair Reduction")
st.caption("Production Planning + Inventory Utilization + Assortment Recommendation")

# =====================================================================
# SIDEBAR
# =====================================================================

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

# OPTIONAL SFG FILE

sfg_file = st.sidebar.file_uploader(
    "SFG_Stock.csv (Optional)",
    type=["csv"]
)

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def allocate_from_locations(size_need: dict, loc_stock_df: pd.DataFrame):

    pool = {}

    for _, row in loc_stock_df.iterrows():

        key = (row['Location'], row['Size'])

        pool[key] = pool.get(key, 0) + int(row['Qty'])

    remaining_need = {
        s: q
        for s, q in size_need.items()
        if q > 0
    }

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

        loc_groups.setdefault(loc, []).append(
            f"Size {size}={qty}"
        )

    allocation_str = " | ".join(

        f"{loc}: {', '.join(parts)}"

        for loc, parts in sorted(loc_groups.items())

    ) if loc_groups else "NA"

    return allocation_str, remaining_need


def format_gap(gap_dict: dict) -> str:

    if not gap_dict:
        return "NA"

    return ", ".join(
        f"Size {s}: +{q}"
        for s, q in sorted(gap_dict.items())
    )

# =====================================================================
# MAIN ENGINE
# =====================================================================

if mapping_file and assort_file and stock_file:

    try:

        with st.spinner("Running Optimization Engine..."):

            # =============================================================
            # LOAD DATA
            # =============================================================

            mapping = pd.read_csv(mapping_file)
            assort_master = pd.read_csv(assort_file)
            stock = pd.read_csv(stock_file)

            # =============================================================
            # OPTIONAL SFG LOAD
            # =============================================================

            sfg_available = False

            if sfg_file is not None:

                sfg_stock = pd.read_csv(sfg_file)

                required_sfg = [
                    'Article Code',
                    'Colour Description',
                    'Assortment Code',
                    'Unrestricted'
                ]

                missing_sfg = [
                    c for c in required_sfg
                    if c not in sfg_stock.columns
                ]

                if missing_sfg:

                    st.error(f"Missing SFG Columns: {missing_sfg}")
                    st.stop()

                sfg_available = True

            # =============================================================
            # VALIDATION
            # =============================================================

            required_mapping = ['Article', 'Colour', 'Assortment']

            required_assort = ['ASST CODE', 'Size', 'Qty']

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

            # =============================================================
            # PREPROCESSING
            # =============================================================

            assort_master['Qty'] = (
                assort_master['Qty']
                .fillna(0)
                .astype(int)
            )

            assort_master['Size'] = (
                assort_master['Size']
                .astype(int)
            )

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

            valid_keys = set(
                mapping['Art_Col'].unique()
            )

            stock = stock[
                stock['Art_Col'].isin(valid_keys)
            ]

            asst_lookup = {

                code: grp.set_index('Size')['Qty'].to_dict()

                for code, grp in assort_master.groupby('ASST CODE')
            }

            # =============================================================
            # KPI METRICS
            # =============================================================

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

            # =============================================================
            # CORE ENGINE
            # =============================================================

            results = []

            groups = list(
                mapping.groupby('Art_Col')
            )

            progress_bar = st.progress(0)

            for i, (art_col, group) in enumerate(groups):

                progress_bar.progress(
                    (i + 1) / len(groups)
                )

                valid_assts = [

                    a

                    for a in group['Assortment'].unique()

                    if a in asst_lookup
                ]

                if not valid_assts:
                    continue

                group_stock = stock[
                    stock['Art_Col'] == art_col
                ]

                stock_map = (
                    group_stock
                    .groupby('Size')['Qty']
                    .sum()
                    .to_dict()
                )

                all_sizes = sorted(

                    list(

                        set(stock_map.keys())

                        |

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

                # =========================================================
                # MILP
                # =========================================================

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

                opportunity_pool = b_u - used_stock

                loc_pool_df = group_stock[
                    ['Location', 'Size', 'Qty']
                ].copy()

                loc_pool_df['Qty'] = (
                    loc_pool_df['Qty']
                    .astype(float)
                )

                for j, asst in enumerate(valid_assts):

                    packs = int(current_packs[j])

                    if packs == 0:
                        continue

                    for size, req_per_pack in asst_lookup[asst].items():

                        total_drain = packs * req_per_pack

                        mask = (
                            loc_pool_df['Size'] == size
                        )

                        for idx2 in loc_pool_df[mask].index:

                            avail = loc_pool_df.at[idx2, 'Qty']

                            take = min(avail, total_drain)

                            loc_pool_df.at[idx2, 'Qty'] -= take

                            total_drain -= take

                            if total_drain <= 0:
                                break

                alt_loc_pool_df = loc_pool_df.copy()

                candidates = []

                for j, asst in enumerate(valid_assts):

                    p_size = sum(
                        asst_lookup[asst].values()
                    )

                    gap = sum(

                        max(
                            0,
                            asst_lookup[asst][s]
                            - optimized_pool[size_idx[s]]
                        )

                        for s in asst_lookup[asst]
                    )

                    candidates.append({

                        'asst': asst,

                        'idx': j,

                        'p_size': p_size,

                        'eff': gap / p_size if p_size > 0 else 1
                    })

                candidates = sorted(
                    candidates,
                    key=lambda x: x['eff']
                )

                for cand in candidates:

                    asst = cand['asst']

                    idx = cand['idx']

                    p_size = cand['p_size']

                    asst_need = asst_lookup[asst]

                    # =====================================================
                    # COMPLETE
                    # =====================================================

                    if current_packs[idx] > 0:

                        n_packs = int(current_packs[idx])

                        total_need = {

                            s: q * n_packs

                            for s, q in asst_need.items()
                        }

                        alloc_str, _ = allocate_from_locations(
                            total_need,
                            group_stock
                        )

                        # OPTIONAL SFG CHECK

                        sfg_value = "NA"

                        if sfg_available:

                            article = art_col.split('_')[0]

                            colour = art_col.split('_')[1]

                            matched_sfg = sfg_stock[

                                (
                                    sfg_stock['Article Code']
                                    .astype(str)
                                    == str(article)
                                )

                                &

                                (
                                    sfg_stock['Colour Description']
                                    .astype(str)
                                    == str(colour)
                                )

                                &

                                (
                                    sfg_stock['Assortment Code']
                                    .astype(str)
                                    == str(asst)
                                )
                            ]

                            if not matched_sfg.empty:

                                sfg_value = int(

                                    matched_sfg['Unrestricted']
                                    .fillna(0)
                                    .sum()
                                )

                        results.append({

                            'Article':
                                art_col.split('_')[0],

                            'Colour':
                                art_col.split('_')[1],

                            'Assortment':
                                asst,

                            'Pack Size':
                                p_size,

                            'Current Packs':
                                n_packs,

                            'Gap Qty':
                                0,

                            'Efficiency':
                                0.0,

                            'Status':
                                'Complete',

                            'Recommendation':
                                f'Fully Packed from existing stock ({n_packs} pack/s)',

                            'Produce (Size:Qty)':
                                'NA',

                            'Available SFG Stock':
                                sfg_value,

                            'Stock Allocation':
                                alloc_str,
                        })

                        continue

                    # =====================================================
                    # GAPS
                    # =====================================================

                    opt_gap = {}

                    for size, req in asst_need.items():

                        avail = optimized_pool[
                            size_idx[size]
                        ]

                        if avail < req:

                            opt_gap[size] = int(
                                req - avail
                            )

                    alt_gap = {}

                    for size, req in asst_need.items():

                        avail = opportunity_pool[
                            size_idx[size]
                        ]

                        if avail < req:

                            alt_gap[size] = int(
                                req - avail
                            )

                    opt_total = sum(opt_gap.values())

                    alt_total = sum(alt_gap.values())

                    # =====================================================
                    # BEST OPTION
                    # =====================================================

                    if opt_total == alt_total:

                        available_need = {

                            s: q

                            for s, q in asst_need.items()

                            if s not in opt_gap
                        }

                        for size, req in asst_need.items():

                            partial = int(

                                min(
                                    optimized_pool[size_idx[size]],
                                    req
                                )
                            )

                            if partial > 0:

                                available_need[size] = partial

                        alloc_str, _ = allocate_from_locations(
                            available_need,
                            loc_pool_df
                        )

                        produce_str = (
                            format_gap(opt_gap)
                            if opt_gap else 'NA'
                        )

                        # OPTIONAL SFG CHECK

                        sfg_value = "NA"

                        if sfg_available:

                            article = art_col.split('_')[0]

                            colour = art_col.split('_')[1]

                            matched_sfg = sfg_stock[

                                (
                                    sfg_stock['Article Code']
                                    .astype(str)
                                    == str(article)
                                )

                                &

                                (
                                    sfg_stock['Colour Description']
                                    .astype(str)
                                    == str(colour)
                                )

                                &

                                (
                                    sfg_stock['Assortment Code']
                                    .astype(str)
                                    == str(asst)
                                )
                            ]

                            if not matched_sfg.empty:

                                sfg_value = int(

                                    matched_sfg['Unrestricted']
                                    .fillna(0)
                                    .sum()
                                )

                        results.append({

                            'Article':
                                art_col.split('_')[0],

                            'Colour':
                                art_col.split('_')[1],

                            'Assortment':
                                asst,

                            'Pack Size':
                                p_size,

                            'Current Packs':
                                0,

                            'Gap Qty':
                                opt_total,

                            'Efficiency':
                                round(opt_total / p_size, 2),

                            'Status':
                                'Best Option',

                            'Recommendation':
                                f'BEST OPTION: Produce {opt_total} pairs to unlock 1 pack',

                            'Produce (Size:Qty)':
                                produce_str,

                            'Available SFG Stock':
                                sfg_value,

                            'Stock Allocation':
                                alloc_str,
                        })

                        for size, req in asst_need.items():

                            avail = optimized_pool[
                                size_idx[size]
                            ]

                            take = min(avail, req)

                            optimized_pool[size_idx[size]] -= take

                            remaining_drain = take

                            mask = (
                                loc_pool_df['Size'] == size
                            )

                            for idx2 in loc_pool_df[mask].index:

                                av2 = loc_pool_df.at[idx2, 'Qty']

                                t2 = min(av2, remaining_drain)

                                loc_pool_df.at[idx2, 'Qty'] -= t2

                                remaining_drain -= t2

                                if remaining_drain <= 0:
                                    break

                    # =====================================================
                    # ALTERNATIVE
                    # =====================================================

                    else:

                        available_need = {}

                        for size, req in asst_need.items():

                            partial = int(

                                min(
                                    opportunity_pool[size_idx[size]],
                                    req
                                )
                            )

                            if partial > 0:

                                available_need[size] = partial

                        alloc_str, _ = allocate_from_locations(
                            available_need,
                            alt_loc_pool_df
                        )

                        produce_str = (
                            format_gap(alt_gap)
                            if alt_gap else 'NA'
                        )

                        # OPTIONAL SFG CHECK

                        sfg_value = "NA"

                        if sfg_available:

                            article = art_col.split('_')[0]

                            colour = art_col.split('_')[1]

                            matched_sfg = sfg_stock[

                                (
                                    sfg_stock['Article Code']
                                    .astype(str)
                                    == str(article)
                                )

                                &

                                (
                                    sfg_stock['Colour Description']
                                    .astype(str)
                                    == str(colour)
                                )

                                &

                                (
                                    sfg_stock['Assortment Code']
                                    .astype(str)
                                    == str(asst)
                                )
                            ]

                            if not matched_sfg.empty:

                                sfg_value = int(

                                    matched_sfg['Unrestricted']
                                    .fillna(0)
                                    .sum()
                                )

                        results.append({

                            'Article':
                                art_col.split('_')[0],

                            'Colour':
                                art_col.split('_')[1],

                            'Assortment':
                                asst,

                            'Pack Size':
                                p_size,

                            'Current Packs':
                                0,

                            'Gap Qty':
                                alt_total,

                            'Efficiency':
                                round(alt_total / p_size, 2),

                            'Status':
                                'Alternative Choice',

                            'Recommendation':
                                f'ALTERNATIVE: Produce {alt_total} pairs if Best Option skipped',

                            'Produce (Size:Qty)':
                                produce_str,

                            'Available SFG Stock':
                                sfg_value,

                            'Stock Allocation':
                                alloc_str,
                        })

            # =============================================================
            # OUTPUT
            # =============================================================

            final_df = pd.DataFrame(results)

            st.success("Optimization Completed Successfully")

            # =============================================================
            # DASHBOARD
            # =============================================================

            st.subheader("Operational Dashboard")

            total_inventory = stock['Qty'].sum()

            complete_df = final_df[
                final_df['Status'] == 'Complete'
            ]

            best_df = final_df[
                (final_df['Status'] == 'Best Option')
                &
                (final_df['Efficiency'] <= 0.2)
            ]

            complete_packs = (
                complete_df['Current Packs']
                .sum()
            )

            complete_pairs_consumed = (

                complete_df['Current Packs']
                * complete_df['Pack Size']

            ).sum()

            best_unlock_pairs = (
                best_df['Pack Size']
                .sum()
            )

            best_gap_qty = (
                best_df['Gap Qty']
                .sum()
            )

            best_instances = len(best_df)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Inventory",
                f"{int(total_inventory):,} Pairs"
            )

            col2.metric(
                "Complete Packs Possible",
                f"{int(complete_packs):,}"
            )

            col3.metric(
                "Pairs Consumed (Complete)",
                f"{int(complete_pairs_consumed):,}"
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Best Option Instances",
                f"{int(best_instances):,}"
            )

            col5.metric(
                "Production Needed",
                f"{int(best_gap_qty):,} Pairs"
            )

            col6.metric(
                "Additional Pairs Unlockable",
                f"{int(best_unlock_pairs):,} Pairs"
            )

            # =============================================================
            # FILTERS
            # =============================================================

            st.subheader("Filters")

            status_filter = st.multiselect(
                "Select Status",
                options=final_df['Status'].unique(),
                default=final_df['Status'].unique()
            )

            filtered_df = final_df[
                final_df['Status'].isin(status_filter)
            ]

            # =============================================================
            # TABLE
            # =============================================================

            st.subheader("Optimization Results")

            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=600
            )

            # =============================================================
            # DOWNLOAD BUTTON
            # =============================================================

            csv = (
                filtered_df
                .to_csv(index=False)
                .encode('utf-8')
            )

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
        "Please upload all 3 CSV files from the sidebar to begin analysis."
    )
