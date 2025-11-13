# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.title("B2B Transaction Dashboard")

# Google Sheets linki
file_id = "1gqQkRnmtTu01gIHWB1yhENb8SO8sWNuz"
downloaded_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"

# Excel dosyasını oku
df = pd.read_excel(downloaded_url)

# Veriyi Streamlit tablosunda göster
st.write("Veri tablosu:")
st.dataframe(df)

# Örnek: Plotly grafiği
if 'Amount' in df.columns and 'Quantity' in df.columns:
    fig = px.scatter(df, x='Quantity', y='Amount', title="Miktar vs Tutar")
    st.plotly_chart(fig)
st.set_page_config(page_title="B2B Product Dashboard", layout="wide", page_icon="📚")

st.title("B2B Book Distributor — Smart Dashboard")

# -------------------------
# Helpers & Caching
# -------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare(filepath="B2B_Transaction_Data.xlsx"):
    """
    Reads the excel and returns:
    - df: original transactions with Sales Revenue and month
    - df_monthly: monthly aggregated sales per StockCode with a 'month' column (1..12)
    - df_final: SKU-level summary with total_revenue, average_sales, std_dev, CV, ABC_Class, XYZ_Class, stock_class, Description
    """
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        raise FileNotFoundError(f"Excel okunamadı: {e}")

    # Basic cleaning: ensure expected columns exist
    expected = {'StockCode','Description','Quantity','UnitPrice','InvoiceDate'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Excel'de beklenen sütun(lar) yok: {missing}")

    # Sales Revenue and month
    df['Sales Revenue'] = df['Quantity'] * df['UnitPrice']
    # ensure InvoiceDate is datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    df['month'] = df['InvoiceDate'].dt.month

    # monthly sales per StockCode
    df_agg = df.groupby(['StockCode','month'])['Sales Revenue'].sum().reset_index()
    df_monthly = df_agg.copy()

    # pivot to get 1..12 columns; keep month sums as 1..12
    pivot = df_agg.pivot(index='StockCode', columns='month', values='Sales Revenue').fillna(0)
    # ensure columns 1..12 present
    for m in range(1,13):
        if m not in pivot.columns:
            pivot[m] = 0
    pivot = pivot.reindex(columns=range(1,13))
    pivot = pivot.reset_index()

    # SKU level summary
    pivot['total_sales'] = pivot.loc[:, 1:12].sum(axis=1)
    pivot['average_sales'] = pivot['total_sales'] / 12
    pivot['std_dev'] = pivot.loc[:, 1:12].std(axis=1)
    # avoid division by zero
    pivot['CV'] = pivot['std_dev'] / pivot['average_sales'].replace(0, np.nan)
    pivot['CV'] = pivot['CV'].fillna(0)

    # XYZ classification based on CV
    def xyz_cv(c):
        if c <= 0.5:
            return 'x'
        elif c <= 1.0:
            return 'y'
        else:
            return 'z'
    pivot['XYZ_Class'] = pivot['CV'].apply(xyz_cv)

    # total revenue (same as total_sales here) but preserve naming
    df_sku = pivot[['StockCode','total_sales','average_sales','std_dev','CV','XYZ_Class']].copy()
    df_4 = df_sku[['StockCode','total_sales']].rename(columns={'total_sales':'total_revenue'}).sort_values('total_revenue', ascending=False).reset_index(drop=True)

    # cumulative percentage for ABC (Pareto)
    df_4['cumulative'] = df_4['total_revenue'].cumsum()
    df_4['total_cumulative'] = df_4['total_revenue'].sum()
    df_4['sku_percentage'] = df_4['cumulative'] / df_4['total_cumulative']

    def abc_pct(x):
        if x > 0 and x <= 0.80:
            return 'A'
        elif x > 0.80 and x <= 0.95:
            return 'B'
        else:
            return 'C'
    df_4['ABC_Class'] = df_4['sku_percentage'].apply(abc_pct)

    # merge SKU features and description
    df_final = pd.merge(df_4, df_sku, on='StockCode', how='left')
    # bring Description
    desc = df[['StockCode','Description']].drop_duplicates()
    df_final = pd.merge(df_final, desc, on='StockCode', how='left')

    # stock_class as combination
    df_final['stock_class'] = df_final['ABC_Class'] + df_final['XYZ_Class'].astype(str)

    # also create df_monthly_sales in longer form (month, StockCode, Sales Revenue)
    df_monthly_sales = df_agg.copy()

    # rename some numeric columns for clarity
    df_final = df_final.rename(columns={'total_revenue':'total_revenue','total_sales':'total_sales'})

    return df, df_monthly_sales, df_final

# Load data (catch errors and show them in Streamlit)
with st.spinner("Veri yükleniyor ve hazırlanıyor..."):
    try:
        df, df_monthly_sales, df_final = load_and_prepare("B2B_Transaction_Data.xlsx")
    except Exception as e:
        st.error(f"Veri yüklenirken hata: {e}")
        st.stop()

# -------------------------
# Sidebar filters (global)
# -------------------------
st.sidebar.header("Filtreler (Global)")
abc_opts = sorted(df_final['ABC_Class'].dropna().unique().tolist())
xyz_opts = sorted(df_final['XYZ_Class'].dropna().unique().tolist())
stockclass_opts = sorted(df_final['stock_class'].dropna().unique().tolist())

abc_selected = st.sidebar.multiselect("ABC Class", abc_opts, default=abc_opts)
xyz_selected = st.sidebar.multiselect("XYZ Class", xyz_opts, default=xyz_opts)
stockclass_selected = st.sidebar.multiselect("Stock Class", stockclass_opts, default=stockclass_opts)

# Page navigation
page = st.sidebar.radio("Sayfalar", ["Overview","KPI Analysis","Analytical Insights","Business Insights"])

# Apply filters to df_final and keep list of StockCodes to filter monthly
df_filtered = df_final.copy()
if abc_selected:
    df_filtered = df_filtered[df_filtered['ABC_Class'].isin(abc_selected)]
if xyz_selected:
    df_filtered = df_filtered[df_filtered['XYZ_Class'].isin(xyz_selected)]
if stockclass_selected:
    df_filtered = df_filtered[df_filtered['stock_class'].isin(stockclass_selected)]

filtered_stockcodes = df_filtered['StockCode'].unique().tolist()

# -------------------------
# Overview Page
# -------------------------
if page == "Overview":
    st.header("Overview")
    st.markdown("Kısa veri özeti ve örnek kayıtlar.")

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("#### Genel KPI'lar (Seçili filtre uygulanmış)")
        total_rev = df_filtered['total_revenue'].sum()
        avg_sales = df_filtered['average_sales'].mean()
        mean_cv = df_filtered['CV'].mean()

        st.metric("Toplam Gelir (Total Revenue)", f"{total_rev:,.2f}")
        st.metric("Ortalama Aylık Satış (SKU başına)", f"{avg_sales:,.2f}")
        st.metric("Ortalama CV", f"{mean_cv:.3f}")

        st.markdown("#### Top 10 Ürün (Gelire Göre)")
        top10 = df_filtered.nlargest(10,'total_revenue')[['Description','total_revenue']]
        st.dataframe(top10.reset_index(drop=True), height=300)

    with col2:
        st.markdown("#### ABC Dağılımı")
        abc_counts = df_filtered['ABC_Class'].value_counts().reset_index()
        abc_counts.columns = ['ABC_Class','Count']
        if not abc_counts.empty:
            fig = px.pie(abc_counts, values='Count', names='ABC_Class', title='ABC Class Distribution')
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### XYZ Dağılımı")
        xyz_counts = df_filtered['XYZ_Class'].value_counts().reset_index()
        xyz_counts.columns = ['XYZ_Class','Count']
        if not xyz_counts.empty:
            fig2 = px.pie(xyz_counts, values='Count', names='XYZ_Class', title='XYZ Class Distribution')
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Seçili Veri Örneği")
    st.dataframe(df_filtered.head(20))

# -------------------------
# KPI Analysis Page
# -------------------------
elif page == "KPI Analysis":
    st.header("KPI Analysis")
    st.markdown("Aylık satış trendleri ve KPI'lar")

    if len(filtered_stockcodes) == 0:
        st.info("Seçime uyan SKU bulunamadı.")
    else:
        # monthly aggregation from df_monthly_sales filtered by stockcodes
        monthly_filtered = df_monthly_sales[df_monthly_sales['StockCode'].isin(filtered_stockcodes)].groupby('month').agg(Total_Sales=('Sales Revenue','sum')).reset_index()
        monthly_filtered = monthly_filtered.sort_values('month')

        if not monthly_filtered.empty:
            fig = px.bar(monthly_filtered, x='month', y='Total_Sales', labels={'month':'Ay','Total_Sales':'Toplam Satış'}, title='Aylık Toplam Satış (Filtreli)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aylık satış verisi yok.")

        st.markdown("#### KPI Panel")
        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam Gelir", f"{df_filtered['total_revenue'].sum():,.2f}")
        col2.metric("Ortalama SKU Geliri", f"{df_filtered['total_revenue'].mean():,.2f}")
        col3.metric("SKU Sayısı", f"{len(df_filtered):,}")

# -------------------------
# Analytical Insights
# -------------------------
elif page == "Analytical Insights":
    st.header("Analytical Insights")
    st.markdown("Derinlemesine analizler: Sales vs CV, Korelasyon, Regresyon")

    if df_filtered.empty:
        st.info("Seçime uyan veri yok.")
    else:
        # scatter avg_sales vs CV
        fig = px.scatter(df_filtered, x='average_sales', y='CV', color='stock_class', hover_name='Description', title='Average Sales vs CV', labels={'average_sales':'Ortalama Aylık Satış','CV':'CV'})
        st.plotly_chart(fig, use_container_width=True)

        # correlation heatmap
        numeric_cols = ['total_revenue','total_sales','average_sales','std_dev','CV','sku_percentage']
        corr_df = df_filtered[numeric_cols].apply(pd.to_numeric, errors='coerce').corr()
        fig2 = px.imshow(corr_df, text_auto=True, title='Korelasyon Matrisi')
        st.plotly_chart(fig2, use_container_width=True)

        # simple regression: total_sales ~ average_sales + std_dev
        st.markdown("#### Basit Lineer Regresyon (total_sales ~ average_sales + std_dev)")
        reg_df = df_filtered[['average_sales','std_dev','total_sales']].dropna()
        if len(reg_df) >= 5:
            X = reg_df[['average_sales','std_dev']].values
            y = reg_df['total_sales'].values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            model = LinearRegression()
            model.fit(Xs,y)
            r2 = model.score(Xs,y)
            coefs = model.coef_
            st.write(f"R-squared: {r2:.3f}")
            st.write(f"Coefficients (scaled): average_sales={coefs[0]:.3f}, std_dev={coefs[1]:.3f}")
        else:
            st.info("Regresyon için yeterli veri yok (min 5 kayıt).")

# -------------------------
# Business Insights
# -------------------------
elif page == "Business Insights":
    st.header("Business Insights")
    st.markdown("Top performing products & outlier analysis")

    if df_filtered.empty:
        st.info("Seçime uyan veri yok.")
    else:
        st.markdown("#### Top 10 Ürün (Toplam Gelire Göre)")
        top10 = df_filtered.nlargest(10,'total_revenue')[['Description','total_revenue','ABC_Class','XYZ_Class','stock_class']]
        st.dataframe(top10.reset_index(drop=True), height=300)
        fig = px.bar(top10, x='total_revenue', y='Description', orientation='h', title='Top 10 Ürün', labels={'total_revenue':'Total Revenue','Description':'Ürün'}, color='total_revenue')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Outlier Analysis (Z-Score on total_revenue)")
        revenue = pd.to_numeric(df_filtered['total_revenue'], errors='coerce').dropna()
        if len(revenue) >= 1:
            z = np.abs(zscore(revenue))
            df_out = df_filtered.copy().loc[revenue.index]
            df_out['zscore'] = z
            outliers = df_out[df_out['zscore'] > 3].sort_values('zscore', ascending=False)
            st.write(f"Toplam Aykırı Ürün Sayısı (Z>3): {len(outliers)}")
            if not outliers.empty:
                st.dataframe(outliers[['Description','total_revenue','zscore']].head(20))
            else:
                st.info("Belirgin aykırı ürün bulunamadı (Z>3).")
        else:
            st.info("Aykırı analiz için yeterli gelir verisi yok.")
