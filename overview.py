import streamlit as st

st.title("Overview")
st.write("This is adopted from https://github.com/fastai/fastbook/tree/master")
st.write("Please wait while it is loading as the training is taking a bit of time.")


def main():
    st.header("List of Pages")
    st.write("1. [Cats or Dogs](/pages/cats-or-dogs)")

if __name__ == "__main__":
    main()
