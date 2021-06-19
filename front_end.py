"""
You need to run the app from the root
To run the app
$ streamlit run serving/model_as_service/streamlit/front_end.py
"""

from pandas.core.frame import DataFrame
import streamlit as st
import pandas as pd
import requests
import json
import ast
import base64

# Found at http://awesome-streamlit.org/ --> Gallery --> Select the App: File Download workaround


def get_table_download_link(df: DataFrame) -> str:
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="batch_predictions">Download prediction as CSV file</a>\
             (right-click and save as &lt;some_name&gt;.csv)'
    return href


def pd_to_json(dataframe: DataFrame):
    df = dataframe.rename(str.lower, axis='columns')
    nb_rows_missing_values = df[df.isnull().any(axis=1)].shape[0]
    df = df.dropna()
    nb_valid_rows = df.shape[0]
    json_file = df.to_json(orient="records")
    parsed = json.loads(json_file)
    return nb_rows_missing_values, nb_valid_rows, json.dumps(parsed)


def print_config(config_dict: dict):
    out = f'''### Model Info  \n**Name**: {config_dict['model_name']}  
            **Model Type**: {config_dict['model_type']}  
            **Filename**: {config_dict['model_filename']}  
            **Version**: {config_dict['model_version']}'''
    return out


st.title("Diabetes prediction tool")

config_button = st.button(label="Show model configuration")
if config_button:
    res = requests.get('https://fastapi-diabetes.herokuapp.com/model_config')
    res_dict = ast.literal_eval(res.text)
    st.markdown(print_config(res_dict))

# Based on first record to avoid long filling
DEFAULT_VALUES = {'age': 59.0,
                  'sex': 2.0,
                  'bmi': 32.1,
                  'bp': 101.0,
                  's1': 157.0,
                  's2': 93.2,
                  's3': 38.0,
                  's4': 4.0,
                  's5': 4.8598,
                  's6': 87.0}

# model = joblib.load("models/diabetes_model.pkl")

# Single prediction form
with st.form(key="predict_form"):
    inputs = {}
    inputs["age"] = st.number_input(label="age", min_value=0,
                                    max_value=150, step=1, key=10, value=int(DEFAULT_VALUES['age']))
    inputs["sex"] = st.selectbox(label="sex (1 for male, 2 for female)",
                                 options=[1, 2], key=11)
    cols = st.beta_columns(2,)

    for i, col in enumerate(cols):
        for j in range(4):
            in_name = list(DEFAULT_VALUES.keys())[2:][j*2+i]
            inputs[in_name] = col.number_input(label=in_name, key=j*2+i, value=DEFAULT_VALUES[in_name])
    submit_button = st.form_submit_button(label='Get prediction')

if submit_button:
    m = {k: inputs[k] for k in DEFAULT_VALUES.keys()}
    res = requests.post('https://fastapi-diabetes.herokuapp.com/predict', data=json.dumps(m))
    st.markdown(f"### Model prediction {float(res.text):.2f}")

"""
# Perform batch predictions from a CSV file:
"""

# import CSV
csv_file = st.file_uploader('Import several patients records with a CSV file')
if csv_file:
    # st.write("filename:", csv_file.name)
    df = pd.read_csv(csv_file, delimiter=";")
    """
    ## First five rows of your CSV to ensure it is the correct one before processing.
    """
    st.write(df.head())
    nb_rejected_rows, nb_valid_rows, diabetes_data_json = pd_to_json(df)
    """
    ## Please click "Predict Diabete evolution" button below to launch predictions:
    """

# Model predictions
if st.button('Predict Diabete evolution'):
    if csv_file:
        # does not work because I need to use the full inference pipeline
        predictions = requests.post(
            'https://fastapi-diabetes.herokuapp.com/predict_obj', data=diabetes_data_json)
        st.write(
            f"The CSV file had {nb_rejected_rows} with missing data, {nb_valid_rows}\
                patients data have been processed properly")
        json_file = ast.literal_eval(predictions.text)
        result_df = pd.DataFrame.from_dict(json_file, orient='index', columns=[
                                           'Diabetes progression']).reset_index(level=0)
        result_df = result_df.rename(columns={"index": "Patient"}, errors="raise")
        st.write(result_df)
        st.markdown(get_table_download_link(result_df), unsafe_allow_html=True)
    else:
        st.warning('You need to upload the CSV')
