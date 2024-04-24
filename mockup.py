import streamlit as st
import numpy as np
import pandas as pd
import time

# Title
st.title("My Streamlit App")

# Header
st.header("Welcome to my app!")

# Subheader
st.subheader("Here are some features:")

# Text
st.write("This is some text.")

# Markdown
st.markdown("## This is a markdown heading")

# Button
if st.button("Click me"):
    st.write("Button clicked!")

# Checkbox
checkbox_state = st.checkbox("Check me")
if checkbox_state:
    st.write("Checkbox checked!")

# Radio buttons
radio_button = st.radio("Choose an option", ("Option 1", "Option 2", "Option 3"))
st.write("Selected option:", radio_button)

# Selectbox
selectbox_option = st.selectbox("Choose an option", ("Option 1", "Option 2", "Option 3"))
st.write("Selected option:", selectbox_option)

# Slider
slider_value = st.slider("Choose a value", 0, 10)
st.write("Selected value:", slider_value)

# Text input
text_input = st.text_input("Enter some text")
st.write("Entered text:", text_input)

# File uploader
uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    st.write("File uploaded!")

# Plotting
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)

st.pyplot(fig)

# Dataframe

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "London", "Paris"]
}

df = pd.DataFrame(data)
st.dataframe(df)

# Table
st.table(df)

# Sidebar
st.sidebar.title("Sidebar")
st.sidebar.write("This is a sidebar.")

# Expander
with st.expander("Click to expand"):
    st.write("This is some hidden content.")

# Footer
st.footer("This is the footer.")

# Show code
with st.echo():
    st.write("This is the code.")

# Show JSON
st.json({"name": "John", "age": 30})

# Show progress

progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.1)

# Show success message
st.success("Task completed successfully!")

# Show error message
st.error("An error occurred.")

# Show warning message
st.warning("This is a warning.")

# Show info message
st.info("This is an info message.")