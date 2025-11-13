import streamlit as st

st.set_page_config(
    page_title="Product Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "KPI Analysis", "Analytical Insights", "Business Insights"])

if page == "Overview":
    st.title("Overview Page")
    st.write("This page will provide a general overview of the product data with interactive filters.")
    import pages.overview as overview
    overview.app()
elif page == "KPI Analysis":
    st.title("KPI Analysis Page")
    st.write("This page will display key performance indicators and relevant charts.")
    import pages.kpi_analysis as kpi_analysis
    kpi_analysis.app()
elif page == "Analytical Insights":
    st.title("Analytical Insights Page")
    st.write("This page will present deeper analytical insights with advanced visualizations.")
    import pages.analytical_insights as analytical_insights
    analytical_insights.app()
elif page == "Business Insights":
    st.title("Business Insights Page")
    st.write("This page will focus on actionable business insights and strategic visualizations.")
    import pages.business_insights as business_insights
    business_insights.app()
