# Experiment Workflow

# Pycaret:Train Machine Learning Pipeline
# Streamlit:Build front-end web app
# Github:Host repository、source code
# HEROKU:PaaS to host web endpoint for ML

from pycaret.regression import *

data = ...

r2 = setup(data=data, target='charges', session_id=123, normalize=True,
           polynomial_features=True, bin_numeric_features=['age', 'bmi'])
lr = create_model('lr')
plot_model(lr, plot='residuals')
save_model(lr, model_name='deployment')


from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('deployment')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('logo.png')
    image_hospital = Image.open('hosptial.jpg')

    st.image(image, use_column_width=False)
    add_selectbox = st.sidebar.selectbox('How would you like to predict?', ('Online', 'Batch'))
    st.sidebar.info('This app is created to predict patient hospital charges')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_hospital)
    
    
    st.title('Insurance Charges Prediction APP')

    if add_selectbox == 'Online':
        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox('Smoker'):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output = ''
        input_dict = {'age':age, 'sex':sex, 'bmi':bmi, 'children':children, 'smoker':smoker, 'region':region}
        input_df = pd.DataFrame(input_dict)

        if st.button('predict'):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader('Upload csv file for predictions', type=['csv'])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)


if __name__ == '__main__':
    run()


# 1. 准备项目文件上传到github仓库

# your_app.py
# requirements.txt
# model.pkl
# setup.sh
'''
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
'''
# Procfile: 'web: sh setup.sh && streamlit run app.py'
# README.md

# 2. 
# a. Sign up on heroku.com and click on ‘Create new app’
# b. Enter App name and region
# c. Connect to your GitHub repository
# d. Deploy branch
# e. Wait 10 minutes and App is published to URL


