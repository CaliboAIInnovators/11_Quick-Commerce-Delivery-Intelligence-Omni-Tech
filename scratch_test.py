import streamlit as st

# Let's inspect what st.html does
from streamlit.proto.Html_pb2 import Html as HtmlProto

html_text = """
<div class="content-box">
    <h3>Hello</h3>
    <p>World</p>
</div>
"""
print("Running test...")
