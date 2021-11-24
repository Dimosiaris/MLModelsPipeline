import streamlit as st
import ModelCreator as mc
import os
import shutil
model_name = ""
st.title('Automatic Classifier Creator') 
st.text('This app creates an Tensorflow ML Model, based on the search inputs you provide.')
st.text('You just need to tell us with what search terms to search the web with \nand we\'ll do the rest!')
st.text('')
st.text('')

with st.form(key="search_form", clear_on_submit=True):
    search_terms = st.text_input('Please add the comma seperated search terms:')

    search_inputs = [x.strip() for x in search_terms.strip().split(",")]

    st.write(search_inputs)
    if st.form_submit_button('Create'):
        with st.spinner("Please wait while the photos are being dowloaded and the model is trained..."):
            mc.train_model_based_on_inputs(search_inputs=search_inputs)
            st.balloons()
            model_name = "model" 
            for term in search_inputs:
                model_name += "-"
                model_name += term 
            print("Saving the zip file...")
            shutil.make_archive(model_name, 'zip', './models')
            print("Done saving the zip file!")

if os.path.isfile('./' + model_name + ".zip"):
    with open(model_name + ".zip", "rb") as model_file:
        st.download_button(label="Download the model", data=model_file,file_name="model.zip", mime="application/zip")
