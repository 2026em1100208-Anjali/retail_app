from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import gc
import os

gc.collect()
# ------------------------------------------------------------
# BITS Retail | Sales & Profitability Dashboard (2024-2026)
# Place "Dataset for Power BI.csv" in the same folder as app.py
# ------------------------------------------------------------

st.set_page_config(
    page_title="BITS Retail | Sales & Profitability",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light-mode visual system: clean canvas, restrained blue/teal accents, readable cards.
st.markdown("""
<style>
    .stApp {background: #f5f7fb; color: #17243a;}
    [data-testid="stHeader"] {background: rgba(245,247,251,.92);}
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1480px;}
    h1 {color:#142b4a; letter-spacing:-.7px; font-weight:750;}
    h2 {color:#183b63; letter-spacing:-.35px; padding-top:.25rem;}
    h3 {color:#234b70;}
    [data-testid="stMetric"] {
        background:#ffffff; border:1px solid #e2e8f0;
        border-radius:14px; padding:15px 17px;
        box-shadow:0 2px 9px rgba(24,49,83,.045);
    }
    [data-testid="stMetricLabel"] p {font-size:.82rem;color:#64748b;font-weight:600;}
    [data-testid="stMetricValue"] {font-size:1.55rem;color:#17365d;font-weight:750;}
    .section-note {
        background:#eaf1fa; border-left:4px solid #3974b9;
        color:#40546d; padding:11px 15px; border-radius:0 9px 9px 0;
        margin:7px 0 17px 0;
    }
    .insight-card {
        background:#ffffff; border:1px solid #dfe7f0; border-left:4px solid #168a83;
        border-radius:10px; padding:13px 16px; margin:10px 0 16px 0;
        color:#24364b; line-height:1.65; box-shadow:0 2px 8px rgba(24,49,83,.035);
    }
    .rec-card {
        background:#ffffff; border:1px solid #dfe7f0; border-radius:13px;
        padding:16px 18px; min-height:180px; box-shadow:0 3px 12px rgba(24,49,83,.045);
    }
    .small-muted {color:#718096;font-size:.82rem;}
    [data-testid="stSidebar"] {background:#eef3f9;}
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color:#183b63;}
    div[data-testid="stDataFrame"] {border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;}
    .stButton>button, .stDownloadButton>button {border-radius:8px;font-weight:600;}
    hr {border-color:#e2e8f0;}
</style>
""", unsafe_allow_html=True)

BLUE = "#1d4ed8"
GREEN = "#0f9d7a"
ORANGE = "#d97706"
RED = "#dc2626"
YEAR_COLORS = {2024: "#9db4d9", 2025: "#4f78c4", 2026: "#173a8a"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DISCOUNT_BANDS = ["0%", "1-5%", "6-10%", "11-15%", "16-20%", ">20%"]

PLOT_FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color="#52647a")

def polish_chart(fig):
    """Apply consistent light-mode chart styling and legible axis labels."""
    fig.update_layout(
        template="plotly_white",
        font=PLOT_FONT,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        title=dict(x=0.015, xanchor="left", font=dict(size=16, color="#183b63")),
        margin=dict(l=18, r=22, t=58, b=48),
        hoverlabel=dict(bgcolor="#17365d", font=dict(color="#ffffff", size=12)),
        legend=dict(
            bgcolor="rgba(255,255,255,.82)", bordercolor="#e7edf4",
            borderwidth=0, font=dict(size=11, color="#52647a")
        ),
    )
    for axis in ("xaxis", "yaxis", "yaxis2"):
        current = getattr(fig.layout, axis, None)
        if current is not None:
            current.update(
                title_font=dict(size=12, color="#40566f"),
                tickfont=dict(size=11, color="#64748b"),
                showline=True, linecolor="#dce4ed", linewidth=1,
                gridcolor="#edf1f5", zerolinecolor="#cbd5e1",
                automargin=True,
            )
    return fig

def insight_box(html):
    st.markdown(f'<div class="insight-card">{html}</div>', unsafe_allow_html=True)


def money(value):
    value = float(value or 0)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}₹{value / 1_000_000:,.2f}M"
    if value >= 1_000:
        return f"{sign}₹{value / 1_000:,.1f}K"
    return f"{sign}₹{value:,.0f}"

def pct(value, digits=1):
    return f"{value * 100:.{digits}f}%" if pd.notna(value) else "—"

FILTER_KEYS = ["filter_years", "filter_regions", "filter_categories",
               "filter_segments", "filter_date_range"]

def reset_filters():
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)

@st.cache_data
def load_data():

    filepath = os.path.join(os.getcwd(), "data", "cleaned_integrated_dataset.csv")
    raw = pd.read_csv(filepath)

    required = [
        "Order_ID", "Order_Date", "Region", "Quantity", "Unit_Price",
        "Discount_Pct", "Shipping_Cost", "Product_Name", "Category",
        "Subcategory", "Unit_Cost", "Customer_Segment",
    ]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    data = raw.copy()
    parsed_dates = pd.to_datetime(data["Order_Date"], errors="coerce")
    parsed_numeric = data[["Quantity", "Unit_Price", "Discount_Pct",
                           "Shipping_Cost", "Unit_Cost"]].apply(
        pd.to_numeric, errors="coerce"
    )
    source_quality = {
        "Source rows": len(raw),
        "Duplicate order IDs": int(raw["Order_ID"].duplicated().sum()),
        "Missing order IDs": int(raw["Order_ID"].isna().sum()),
        "Invalid/missing dates": int(parsed_dates.isna().sum()),
        "Rows with invalid/missing numeric values": int(parsed_numeric.isna().any(axis=1).sum()),
        "Discounts outside 0-100%": int(
            ((parsed_numeric["Discount_Pct"] < 0) |
             (parsed_numeric["Discount_Pct"] > 1)).sum()
        ),
    }
    data["Order_Date"] = parsed_dates
    numeric_cols = ["Quantity", "Unit_Price", "Discount_Pct",
                    "Shipping_Cost", "Unit_Cost"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Order_Date"] + numeric_cols)

    # Use the standardised category field supplied in your CSV, if present.
    if "Fixed Category" in data.columns:
        data["Category_Clean"] = data["Fixed Category"].fillna(data["Category"])
    else:
        category_map = {
            "Tech": "Technology",
            "Office Suply": "Office Supplies",
            "Office Suppllies": "Office Supplies",
        }
        data["Category_Clean"] = data["Category"].replace(category_map)

    # Calculations follow the definitions in the supplied HTML dashboard:
    # Net Sales = (Quantity x Unit Price) - Discount
    # Profit = Net Sales - COGS - Shipping Cost
    data["Gross_Sales"] = data["Quantity"] * data["Unit_Price"]
    data["Discount_Amount"] = data["Gross_Sales"] * data["Discount_Pct"]
    data["Net_Sales"] = data["Gross_Sales"] - data["Discount_Amount"]
    data["COGS"] = data["Quantity"] * data["Unit_Cost"]
    data["Profit"] = data["Net_Sales"] - data["COGS"] - data["Shipping_Cost"]
    data["Profit_Margin"] = np.where(
        data["Net_Sales"].ne(0), data["Profit"] / data["Net_Sales"], np.nan
    )
    data["Year"] = data["Order_Date"].dt.year
    data["Month_Num"] = data["Order_Date"].dt.month
    data["Month"] = data["Order_Date"].dt.strftime("%b")
    data["Quarter"] = "Q" + data["Order_Date"].dt.quarter.astype(str)

    def discount_band(discount):
        if discount == 0:
            return "0%"
        if discount <= .05:
            return "1-5%"
        if discount <= .10:
            return "6-10%"
        if discount <= .15:
            return "11-15%"
        if discount <= .20:
            return "16-20%"
        return ">20%"

    data["Discount_Band"] = data["Discount_Pct"].apply(discount_band)
    data.attrs["source_quality"] = source_quality
    return data

st.title("BITS Retail | Sales & Profitability")
st.caption("2024-2026  ·  Retail performance overview  ·  Financial values in INR (₹)  ·  Insights respond to every filter")

with st.sidebar:
    st.header("Dashboard filters")
    # uploaded = st.file_uploader("Upload / replace dataset (CSV)", type=["csv"])
    try:
        df = load_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if df is None:
        st.warning("Place 'Dataset for Power BI.csv' beside app.py, or upload it here.")
        st.stop()

    years = sorted(df["Year"].dropna().unique().tolist())
    regions = sorted(df["Region"].dropna().unique().tolist())
    categories = sorted(df["Category_Clean"].dropna().unique().tolist())
    segments = sorted(df["Customer_Segment"].dropna().unique().tolist())

    selected_years = st.multiselect("Year", years, default=years, key="filter_years")
    selected_regions = st.multiselect("Region", regions, default=regions, key="filter_regions")
    selected_categories = st.multiselect(
        "Category", categories, default=categories, key="filter_categories"
    )
    selected_segments = st.multiselect(
        "Customer segment", segments, default=segments, key="filter_segments"
    )

    min_date, max_date = df["Order_Date"].min().date(), df["Order_Date"].max().date()
    date_range = st.date_input(
        "Order date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date, key="filter_date_range"
    )
    st.divider()
    st.button("Reset filters", use_container_width=True, on_click=reset_filters)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

filtered = df[
    df["Year"].isin(selected_years)
    & df["Region"].isin(selected_regions)
    & df["Category_Clean"].isin(selected_categories)
    & df["Customer_Segment"].isin(selected_segments)
    & df["Order_Date"].dt.date.between(start_date, end_date)
].copy()

if filtered.empty:
    st.warning("No orders match the selected filters. Change one or more filters.")
    st.stop()

# Common aggregation
def aggregate(frame, by):
    result = frame.groupby(by, dropna=False).agg(
        Orders=("Order_ID", "nunique"),
        Gross_Sales=("Gross_Sales", "sum"),
        Discount_Amount=("Discount_Amount", "sum"),
        Net_Sales=("Net_Sales", "sum"),
        Profit=("Profit", "sum"),
        COGS=("COGS", "sum"),
        Shipping=("Shipping_Cost", "sum"),
    ).reset_index()
    result["Profit_Margin"] = np.where(
        result["Net_Sales"].ne(0), result["Profit"] / result["Net_Sales"], np.nan
    )
    result["Avg_Order_Value"] = np.where(
        result["Orders"].ne(0), result["Net_Sales"] / result["Orders"], np.nan
    )
    result["Avg_Discount"] = np.where(
        result["Gross_Sales"].ne(0),
        result["Discount_Amount"] / result["Gross_Sales"], np.nan
    )
    return result

total_sales = filtered["Net_Sales"].sum()
total_profit = filtered["Profit"].sum()
total_discount = filtered["Discount_Amount"].sum()
gross_sales = filtered["Gross_Sales"].sum()
loss_orders = int((filtered["Profit"] < 0).sum())
margin = total_profit / total_sales if total_sales else 0
avg_discount = total_discount / gross_sales if gross_sales else 0

st.subheader("Performance at a glance")
kpis = st.columns(6)
kpi_values = [
    ("Net sales", money(total_sales)),
    ("Profit", money(total_profit)),
    ("Profit margin", pct(margin)),
    ("Orders", f"{filtered['Order_ID'].nunique():,}"),
    ("Average discount", pct(avg_discount)),
    ("Loss-making orders", f"{loss_orders:,} ({pct(loss_orders / len(filtered), 0)})"),
]
for col, (label, value) in zip(kpis, kpi_values):
    col.metric(label, value)

st.caption(f"Showing {len(filtered):,} rows from {filtered['Order_Date'].min():%d %b %Y} "
           f"to {filtered['Order_Date'].max():%d %b %Y}. All metrics respond to the filters.")

# ----------------------------- Q1 -----------------------------
st.divider()
st.header("Q1  |  How have sales changed from 2024 to 2026?")
st.markdown('<div class="section-note">Explore monthly sales patterns, annual performance and seasonality.</div>',
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    monthly = (filtered.groupby(["Year", "Month_Num"], as_index=False)
               .agg(Net_Sales=("Net_Sales", "sum")))
    monthly["Month"] = monthly["Month_Num"].map(lambda x: MONTHS[int(x)-1])
    monthly["Year"] = monthly["Year"].astype(str)
    fig = px.line(monthly, x="Month", y="Net_Sales", color="Year",
                  markers=True, category_orders={"Month": MONTHS},
                  color_discrete_map={str(k): v for k, v in YEAR_COLORS.items()},
                  title="Monthly net sales by year (₹)")
    fig.update_layout(legend_title_text="", yaxis_title="Net sales (INR ₹)", xaxis_title="Order month")
    st.plotly_chart(polish_chart(fig), use_container_width=True)

with c2:
    annual = aggregate(filtered, "Year").sort_values("Year")
    fig = go.Figure()
    fig.add_bar(x=annual["Year"].astype(str), y=annual["Net_Sales"], name="Net sales",
                marker_color=BLUE)
    fig.add_bar(x=annual["Year"].astype(str), y=annual["Profit"], name="Profit",
                marker_color=GREEN)
    fig.add_trace(go.Scatter(x=annual["Year"].astype(str),
                             y=annual["Profit_Margin"] * 100,
                             name="Profit margin (%)", mode="lines+markers",
                             yaxis="y2", line=dict(color=ORANGE, width=3)))
    fig.update_layout(title="Annual sales, profit and profit margin",
                      barmode="group", xaxis_title="Order year", yaxis_title="Amount (INR ₹)",
                      yaxis2=dict(title="Margin (%)", overlaying="y", side="right"),
                      legend_title_text="")
    st.plotly_chart(polish_chart(fig), use_container_width=True)

annual = aggregate(filtered, "Year").sort_values("Year")
quarter = aggregate(filtered, "Quarter")
month_rank = filtered.groupby("Month")["Net_Sales"].sum().reindex(MONTHS).dropna()
if len(annual) > 1 and int(annual.iloc[-1]["Year"]) - int(annual.iloc[-2]["Year"]) == 1:
    previous_sales = annual.iloc[-2]["Net_Sales"]
    latest_growth = (
        annual.iloc[-1]["Net_Sales"] / previous_sales - 1
        if previous_sales else np.nan
    )
    growth_text = f"Latest year-on-year sales change: **{pct(latest_growth)}**."
elif len(annual) > 1:
    growth_text = "Year-on-year growth is unavailable because the selected years are not consecutive."
else:
    growth_text = "Select at least two consecutive years to calculate year-on-year growth."
q4_share = filtered.loc[filtered["Quarter"] == "Q4", "Net_Sales"].sum() / total_sales if total_sales else 0
peak_month = month_rank.idxmax() if not month_rank.empty else "—"
quarter_sales = filtered.groupby("Quarter")["Net_Sales"].sum().reindex(["Q1","Q2","Q3","Q4"]).dropna()
best_quarter = quarter_sales.idxmax() if not quarter_sales.empty else "—"
weak_quarter = quarter_sales.idxmin() if not quarter_sales.empty else "—"
quarter_share = quarter_sales / total_sales if total_sales else quarter_sales * 0
insight_box(
    f"<b>Live sales story</b><br>{growth_text} "
    f"<b>{best_quarter}</b> is the strongest quarter ({pct(quarter_share.get(best_quarter, 0), 0)} of sales), "
    f"while <b>{weak_quarter}</b> is the weakest. Q4 contributes <b>{pct(q4_share, 0)}</b> of selected sales. "
    f"The strongest month by sales is <b>{peak_month}</b>. "
    f"These insights recalculate whenever the sidebar filters change."
)

# ----------------------------- Q2 -----------------------------
st.divider()
st.header("Q2  |  Which regions lead and lag?")
st.markdown('<div class="section-note">Compare regional scale with profitability. Higher sales do not automatically mean a higher profit margin.</div>',
            unsafe_allow_html=True)
region = aggregate(filtered, "Region").sort_values("Net_Sales", ascending=False)
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_bar(x=region["Region"], y=region["Net_Sales"], name="Net sales", marker_color=BLUE)
    fig.add_bar(x=region["Region"], y=region["Profit"], name="Profit", marker_color=GREEN)
    fig.update_layout(title="Net sales and profit by region (₹)", barmode="group",
                      xaxis_title="Region", yaxis_title="Amount (INR ₹)", legend_title_text="")
    st.plotly_chart(polish_chart(fig), use_container_width=True)
with c2:
    plot_region = region.melt(id_vars="Region",
                              value_vars=["Profit_Margin", "Avg_Discount"],
                              var_name="Metric", value_name="Rate")
    plot_region["Metric"] = plot_region["Metric"].map(
        {"Profit_Margin": "Profit margin", "Avg_Discount": "Average discount"})
    fig = px.bar(plot_region, x="Region", y="Rate", color="Metric", barmode="group",
                 title="Profit margin vs average discount by region")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Rate (%)", xaxis_title="Region",
                      legend_title_text="")
    st.plotly_chart(polish_chart(fig), use_container_width=True)

if len(region):
    best_sales = region.iloc[0]
    low_sales = region.iloc[-1]
    best_margin = region.loc[region["Profit_Margin"].idxmax()]
    low_margin = region.loc[region["Profit_Margin"].idxmin()]
    best_profit = region.loc[region["Profit"].idxmax()]
    high_discount = region.loc[region["Avg_Discount"].idxmax()]
    insight_box(
        f"<b>Live regional comparison</b><br>Sales leader: <b>{best_sales['Region']} ({money(best_sales['Net_Sales'])})</b>; "
        f"lowest sales: <b>{low_sales['Region']} ({money(low_sales['Net_Sales'])})</b>. "
        f"Profit leader: <b>{best_profit['Region']} ({money(best_profit['Profit'])})</b>. "
        f"Best margin: <b>{best_margin['Region']} ({pct(best_margin['Profit_Margin'])})</b>; "
        f"weakest margin: <b>{low_margin['Region']} ({pct(low_margin['Profit_Margin'])})</b>. "
        f"Highest average discount: <b>{high_discount['Region']} ({pct(high_discount['Avg_Discount'])})</b>. "
        f"Compare the leaders to see whether sales scale is translating into profit."
    )

# ----------------------------- Q3 -----------------------------
st.divider()
st.header("Q3  |  Which categories and subcategories contribute most?")
st.markdown('<div class="section-note">Identify the categories, product groups and customer segments driving sales and profit.</div>',
            unsafe_allow_html=True)
category = aggregate(filtered, "Category_Clean").sort_values("Net_Sales", ascending=False)
subcategory = aggregate(filtered, ["Category_Clean", "Subcategory"]).sort_values("Net_Sales")
segment = aggregate(filtered, "Customer_Segment").sort_values("Net_Sales", ascending=False)

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_bar(x=category["Category_Clean"], y=category["Net_Sales"], name="Net sales",
                marker_color=BLUE)
    fig.add_bar(x=category["Category_Clean"], y=category["Profit"], name="Profit",
                marker_color=GREEN)
    fig.update_layout(title="Net sales and profit by category", barmode="group",
                      xaxis_title="Product category", yaxis_title="Amount (INR ₹)", legend_title_text="")
    st.plotly_chart(polish_chart(fig), use_container_width=True)
with c2:
    subcategory["Label"] = subcategory["Category_Clean"] + " - " + subcategory["Subcategory"]
    fig = go.Figure()
    fig.add_bar(y=subcategory["Label"], x=subcategory["Net_Sales"], name="Net sales",
                orientation="h", marker_color=BLUE)
    fig.add_bar(y=subcategory["Label"], x=subcategory["Profit"], name="Profit",
                orientation="h", marker_color=GREEN)
    fig.update_layout(title="Net sales and profit by subcategory", barmode="group",
                      xaxis_title="Amount (INR ₹)", yaxis_title="Category and subcategory", legend_title_text="",
                      height=max(380, 32 * len(subcategory)))
    st.plotly_chart(polish_chart(fig), use_container_width=True)

fig = go.Figure()
fig.add_bar(x=segment["Customer_Segment"], y=segment["Net_Sales"], name="Net sales",
            marker_color=BLUE)
fig.add_bar(x=segment["Customer_Segment"], y=segment["Profit"], name="Profit",
            marker_color=GREEN)
fig.update_layout(title="Net sales and profit by customer segment", barmode="group",
                  xaxis_title="Customer segment", yaxis_title="Amount (INR ₹)", legend_title_text="")
st.plotly_chart(polish_chart(fig), use_container_width=True)

if not category.empty and not subcategory.empty:
    top_cat = category.iloc[0]
    top_sub = subcategory.iloc[-1]
    low_sub = subcategory.iloc[0]
    top_segment = segment.iloc[0]
    low_segment_margin = segment.loc[segment["Profit_Margin"].idxmin()]
    best_sub_margin = subcategory.loc[subcategory["Profit_Margin"].idxmax()]
    insight_box(
        f"<b>Live product and customer mix</b><br>Top category: <b>{top_cat['Category_Clean']}</b> "
        f"({pct(top_cat['Net_Sales'] / total_sales, 0)} of selected sales). "
        f"Top subcategory: <b>{top_sub['Category_Clean']} - {top_sub['Subcategory']}</b>; "
        f"lowest by sales: <b>{low_sub['Category_Clean']} - {low_sub['Subcategory']}</b>. "
        f"Best subcategory margin: <b>{best_sub_margin['Category_Clean']} - {best_sub_margin['Subcategory']} "
        f"({pct(best_sub_margin['Profit_Margin'])})</b>. "
        f"Largest customer segment by sales: <b>{top_segment['Customer_Segment']}</b>; "
        f"lowest segment margin: <b>{low_segment_margin['Customer_Segment']} "
        f"({pct(low_segment_margin['Profit_Margin'])})</b>."
    )

# ----------------------------- Q4 -----------------------------
st.divider()
st.header("Q4  |  What is the relationship between discounts and profitability?")
st.markdown('<div class="section-note">Review margin by discount band and inspect the order-level relationship between discount and profit margin.</div>',
            unsafe_allow_html=True)
discount = aggregate(filtered, "Discount_Band")
discount["Discount_Band"] = pd.Categorical(discount["Discount_Band"],
                                             categories=DISCOUNT_BANDS, ordered=True)
discount = discount.sort_values("Discount_Band")
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(discount, x="Discount_Band", y="Profit_Margin",
                 title="Profit margin by discount band",
                 color="Profit_Margin", color_continuous_scale=["#dc2626", "#f3c969", "#0f9d7a"])
    fig.update_layout(xaxis_title="Order discount band", yaxis_title="Profit margin (%)",
                      yaxis_tickformat=".0%", coloraxis_showscale=False)
    st.plotly_chart(polish_chart(fig), use_container_width=True)
with c2:
    scatter = filtered.copy()
    scatter["Profit_Margin_Pct"] = scatter["Profit_Margin"] * 100
    scatter["Discount_Pct_Display"] = scatter["Discount_Pct"] * 100
    fig = px.scatter(scatter, x="Discount_Pct_Display", y="Profit_Margin_Pct",
                     color="Category_Clean", hover_data=["Order_ID", "Product_Name",
                     "Net_Sales", "Profit"], title="Each order: discount % vs profit margin %",
                     opacity=.65)
    fig.update_layout(xaxis_title="Discount applied per order (%)", yaxis_title="Order profit margin (%)")
    st.plotly_chart(polish_chart(fig), use_container_width=True)

st.subheader("Discount band summary")
discount_table = discount[["Discount_Band", "Orders", "Net_Sales", "Profit",
                           "Profit_Margin", "Avg_Order_Value"]].copy()
discount_table["Profit_Margin"] = discount_table["Profit_Margin"].map(lambda x: pct(x))
discount_table["Avg_Order_Value"] = discount_table["Avg_Order_Value"].map(money)
discount_table["Net_Sales"] = discount_table["Net_Sales"].map(money)
discount_table["Profit"] = discount_table["Profit"].map(money)
discount_table.columns = ["Discount band", "Orders", "Net sales", "Profit",
                          "Profit margin", "Average order value"]
st.dataframe(discount_table, use_container_width=True, hide_index=True)

if len(filtered) >= 3 and filtered["Discount_Pct"].nunique() > 1 and filtered["Profit_Margin"].nunique() > 1:
    discount_corr = filtered["Discount_Pct"].corr(filtered["Profit_Margin"])
else:
    discount_corr = np.nan
if not discount.empty:
    weakest_discount = discount.loc[discount["Profit_Margin"].idxmin()]
    strongest_discount = discount.loc[discount["Profit_Margin"].idxmax()]
    insight_box(
        f"<b>Live discount and profitability signal</b><br>"
        f"Correlation between order discount and order profit margin: "
        f"<b>{discount_corr:.2f}</b>" if pd.notna(discount_corr) else
        "<b>Live discount and profitability signal</b><br>Not enough variation in the current selection to calculate correlation."
    )
    if not discount.empty:
        st.caption(
            f"Within the selected data, the weakest margin is in the "
            f"{weakest_discount['Discount_Band']} band ({pct(weakest_discount['Profit_Margin'])}); "
            f"the strongest is in the {strongest_discount['Discount_Band']} band "
            f"({pct(strongest_discount['Profit_Margin'])}). Correlation describes association, not causation."
        )

# ----------------------------- Q5 -----------------------------
st.divider()
st.header("Q5  |  Two actions to prioritise")
st.markdown('<div class="section-note">Recommendations are generated from the currently selected data, not fixed statements.</div>',
            unsafe_allow_html=True)

recommendations = []

# Action 1: evaluate a 15% discount guardrail using the selected orders.
high_discount_orders = filtered[filtered["Discount_Pct"] > .15]
high_discount_sales = high_discount_orders["Net_Sales"].sum()
high_discount_profit = high_discount_orders["Profit"].sum()
high_discount_loss_count = int((high_discount_orders["Profit"] < 0).sum())
potential_discount_recovery = (
    high_discount_orders["Gross_Sales"] * (high_discount_orders["Discount_Pct"] - .15)
).sum()
discount_share_sales = high_discount_sales / total_sales if total_sales else 0
discount_share_profit = high_discount_profit / total_profit if total_profit else 0

# Action 2: customer attach-rate opportunity. Keep this scoped to technology buyers.
customer_col = "Customer_ID" if "Customer_ID" in filtered.columns else None
tech_accessories = filtered[
    (filtered["Category_Clean"] == "Technology") &
    (filtered["Subcategory"].astype(str).str.lower() == "accessories")
]
tech_core = filtered[
    (filtered["Category_Clean"] == "Technology") &
    (filtered["Subcategory"].astype(str).str.lower().isin(["computers", "phones"]))
]
accessory_margin = (
    tech_accessories["Profit"].sum() / tech_accessories["Net_Sales"].sum()
    if tech_accessories["Net_Sales"].sum() else np.nan
)
core_margin = (
    tech_core["Profit"].sum() / tech_core["Net_Sales"].sum()
    if tech_core["Net_Sales"].sum() else np.nan
)
if customer_col and not tech_core.empty:
    core_customers = set(tech_core[customer_col].dropna().unique())
    accessory_customers = set(tech_accessories[customer_col].dropna().unique())
    attached_customers = core_customers & accessory_customers
    attach_rate = len(attached_customers) / len(core_customers) if core_customers else np.nan
    attachable_customers = len(core_customers - accessory_customers)
else:
    attach_rate, attachable_customers = np.nan, 0

rec_cols = st.columns(2, border=True)
with rec_cols[0]:
    # st.markdown('<div class="rec-card">', unsafe_allow_html=True)
    st.subheader("1 · Set a discount review threshold")
    if len(high_discount_orders):
        st.markdown(
            f"Orders discounted **above 15%**: **{len(high_discount_orders):,}** "
            f"({pct(len(high_discount_orders) / len(filtered), 1)} of selected orders). "
            f"They contribute **{money(high_discount_sales)}** in net sales "
            f"({pct(discount_share_sales)} of sales) and **{money(high_discount_profit)}** in profit. "
            f"**{high_discount_loss_count}** of these orders are loss-making."
        )
        st.markdown(
            f"Reducing the discount on these orders to 15%, assuming quantities and order volume stay unchanged, "
            f"would recover up to **{money(potential_discount_recovery)}** before any customer response."
        )
        st.caption("This is a scenario estimate, not a forecast. Pilot with a controlled test and monitor conversion.")
    else:
        st.write("No orders above 15% discount appear in the current filter selection. The threshold is a review trigger, not an automatic pricing rule.")
    st.markdown('</div>', unsafe_allow_html=True)

with rec_cols[1]:
    # st.markdown('<div class="rec-card">', unsafe_allow_html=True)
    st.subheader("2 · Improve technology accessory attachment")
    if pd.notna(attach_rate):
        st.markdown(
            f"**{len(core_customers):,}** customers purchased a computer or phone. "
            f"**{len(attached_customers):,}** also purchased a technology accessory "
            f"(attach rate **{pct(attach_rate)}**). "
            f"That leaves **{attachable_customers:,}** core-product customers without a recorded accessory purchase."
        )
        st.markdown(
            f"Technology accessories currently have a **{pct(accessory_margin)}** margin, "
            f"compared with **{pct(core_margin)}** for computers and phones."
        )
        st.caption("Test relevant bundles (for example, laptop + mouse/dock) and measure incremental profit, not just units.")
    else:
        st.write("The current selection does not contain enough computer/phone customer records to calculate an accessory attach rate.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------- Detail / data quality -----------------------------
with st.expander("View filtered order-level data"):
    display_cols = ["Order_ID", "Order_Date", "Region", "Category_Clean",
                    "Subcategory", "Product_Name", "Customer_Segment",
                    "Quantity", "Gross_Sales", "Discount_Amount", "Net_Sales",
                    "COGS", "Shipping_Cost", "Profit", "Profit_Margin"]
    st.dataframe(filtered[display_cols].sort_values("Order_Date", ascending=False),
                 use_container_width=True, hide_index=True)
    csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data as CSV", csv_bytes,
                       file_name="bits_retail_filtered_data.csv", mime="text/csv")

with st.expander("Data definitions and quality checks"):
    st.markdown("""
    - **Gross sales:** Quantity x Unit price.
    - **Discount amount:** Gross sales x Discount percentage.
    - **Net sales:** Gross sales - Discount amount.
    - **COGS:** Quantity x Unit cost.
    - **Profit:** Net sales - COGS - shipping cost.
    - **Profit margin:** Profit ÷ Net sales.
    - Shipping cost is treated as an order-level cost and is not multiplied by quantity.
    - The supplied `Fixed Category` field is used for standardised category labels when available.
    """)
    source_quality = df.attrs.get("source_quality", {})
    quality = pd.DataFrame({
        "Check": [
            "Rows in source file", "Rows retained after required-field cleaning",
            "Unique order IDs in retained data", "Duplicate order IDs in source",
            "Missing order IDs in source", "Invalid/missing dates in source",
            "Rows with invalid/missing numeric values",
            "Discounts outside 0-100% (source)",
        ],
        "Result": [
            f"{source_quality.get('Source rows', len(df)):,}",
            f"{len(df):,}", f"{df['Order_ID'].nunique():,}",
            f"{source_quality.get('Duplicate order IDs', 0):,}",
            f"{source_quality.get('Missing order IDs', 0):,}",
            f"{source_quality.get('Invalid/missing dates', 0):,}",
            f"{source_quality.get('Rows with invalid/missing numeric values', 0):,}",
            f"{source_quality.get('Discounts outside 0-100%', 0):,}",
        ]
    })
    st.dataframe(quality, hide_index=True, use_container_width=True)

st.caption("BITS Retail Dashboard | Built with Streamlit, Pandas and Plotly")
