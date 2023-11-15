import streamlit as st
from stqdm import stqdm
import pandas as pd
import numpy as np
import time


## List of todos / changes
# Sidebar to enter a new schema by pasting JSON
# Rerun all with new schema
# Clear history button
# reorder cells

st.title('Data Formatter')

from main import process_raw_text, load_lora_model, json_schema

from src.utils import make_random_loads #(n_loads, json_schema)


fake_loads = make_random_loads(3, json_schema)
print("\n".join(fake_loads[1]))


json_schema = {
    "type": "object",
    "properties": {
        "origin_city": {
            "type": "string", 
            'description': 'What is the origin city name of this message?'
        },
        "origin_state": {
            "type": "string", 
            'description': 'What is the origin statecode for this message?'
        },
        "destination_city": {
            "type": "string",
            'description': 'What is the destination city name of this message?'
        },
        "destination_state": {
            "type": "string", 
            'description': 'What is the destination statecode for this message?'
        },
        "equipment": {
            "type": "string",
            "enum": ['Van', 'Reefer', 'Flatbed'],
            "description": "What is the type of message often Van, Reefer, or Flatbed?"
        },
    }
}


@st.cache_resource  
def load_model_st():
    st.write("Loading model...")
    starttime = time.time()
    lora_name = "loras/lora_flan_xl_1epoc_c/"
    lora_name = "loras/lora_flan_l_1epoc_b/"
    model, tokenizer = load_lora_model(lora_name)
    st.write(f"Loaded in {time.time() - starttime}s")
    return model, tokenizer


## SETUP - Model and session state

# Get model
model, tokenizer = load_model_st()

# Initialize State
if 'prediction_out' not in st.session_state:
    st.session_state['prediction_out'] = []

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

if 'out_schema' not in st.session_state:
    st.session_state['out_schema'] = json_schema

if 'request_history' not in st.session_state:
    st.session_state['request_history'] = []

## Define callbacks
def pred_requests(request_data):

    for _load_data in stqdm(request_data):
        if len(_load_data.strip()) == 0:
            continue

        st.session_state['request_history'].append(_load_data)

        prediction = process_raw_text(
            model, tokenizer,
            st.session_state['out_schema'], 
            #json_schema,
            _load_data)
        
        st.session_state['prediction_out'].append(prediction)


        st.session_state['user_input'] = ""

def button1_callback():
    if len(load_data) > 0:
        pred_requests(load_data)

def rerun_inputs_callback():

    requests = st.session_state.request_history.copy()
    st.session_state.request_history = []
    st.session_state.prediction_out = []

    pred_requests(requests)
    
def clear_output_callback():
    st.session_state.request_history = []
    st.session_state.prediction_out = []


def add_json_field(name, dtype, description):
    new_item = {
        'type': dtype.lower(),
        'description': description
    }

    st.session_state['out_schema']['properties'][name] = new_item


def get_df_as_csv():
    df = pd.DataFrame(st.session_state['prediction_out'])
    return df.to_csv().encode('utf-8')

def delete_key_from_schema(key):
    print(key)
    print(st.session_state.out_schema['properties'].pop(key))

###### Start of application #######
###################################


st.header("Enter raw input:")
multi_entry = st.checkbox("Seperate outputs by linebreaks", value=True)
load_data = st.text_area(label="Raw text", key='user_input')
if multi_entry:
    load_data = load_data.split('\n')
else:
    load_data = [load_data]


# https://stackoverflow.com/a/77332142
# Trick to make layout of horizontal buttons better
st.markdown("""
            <style>
                div[data-testid="column"] {
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="column"] * {
                    width: fit-content !important;
                }
            </style>
            """, unsafe_allow_html=True)

col0, col1, col2, col3 = st.columns([1, 1,1,1])
with col0:
    st.button("submit", on_click=button1_callback)
with col1:
    st.button('Re-run requests', on_click=rerun_inputs_callback)
with col2:
    st.button('Clear output', on_click=clear_output_callback)
with col3:
    st.download_button(
        label="Download Output",
        data=get_df_as_csv(),
        file_name="load_formatter_output.csv",
        mime="text/csv",
        key='download-csv'
    )


if len(st.session_state['prediction_out']) > 0:
    df_out = pd.DataFrame(st.session_state['prediction_out'])
    st.table(df_out)


# Request History 
st.subheader("Request History")
st.table(pd.DataFrame(st.session_state['request_history']))


st.header("Schema Builder")

field_name = st.text_input(label="Field Name")
field_dtype = st.selectbox(
    'Dtype of field',
    ('Number', 'String', 'Boolean'))
field_desc = st.text_area(
    label="Enter description of field"
)

if st.button(label="Add field"):
    add_json_field(field_name, field_dtype, field_desc)

## Show current items
for key, item in st.session_state['out_schema']['properties'].items():
    col0, col1 = st.columns([1, 1])

    with col0:
        st.write(f"- {key} ({item['type']}) - {item.get('description')}")
    with col1:
        st.button("Delete", key=key, on_click=lambda : delete_key_from_schema(key))



#st.header("Output Schema:")
#out_schema = st.text_area(label="out schema", value=json_schema)

## St sidebar
# Using "with" notation
if False:
    with st.sidebar:
        st.data_editor()
        raw_json = st.text_area(
            label='output json schema',
            value=st.session_state['out_schema'])
