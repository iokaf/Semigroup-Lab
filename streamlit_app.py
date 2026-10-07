"""Semigroup Lab (Streamlit edition): analysis of continuous semigroups of holomorphic
self-maps of the unit disc. Run with:  streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Semigroup Lab", page_icon=":material/blur_circular:", layout="wide")

pages = [
    st.Page("views/analyse.py", title="Analyse", icon=":material/insights:", default=True),
    st.Page("views/docs.py", title="How it works", icon=":material/menu_book:", url_path="docs"),
]
st.navigation(pages, position="top").run()
