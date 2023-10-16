from transformers import pipeline
import transformers
import accelerate
import time
from src.jsonformer import Jsonformer, JsonFormerText2Text
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import pandas as pd
import random
import torch
from src.utils import make_random_loads
import numpy as np


# Text-generation
model_name = 'tiiuae/falcon-7b'
model_name = 'databricks/dolly-v2-12b'
model_name = 'databricks/dolly-v2-3b'

## Text2text
model_type = 'text2text-generation'
model_name = 'google/flan-t5-large'
# model_name = 'google/flan-t5-xxl'
#model_name = "tiiuae/falcon-rw-1b"

#model_name = 'cerebras/Cerebras-GPT-590M'


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
        "weight": {
            "type": "number", "description":
            "Get the weight as a number from this message"
                   },
        "weight_unit": {"type": "string"},
        "hot": {
            "type": "boolean",
            "default": False,
            "description": "True or False Does this have the message HOT in it?",
        },
        "team": {
            "type": "boolean",
            "default": False,
            "description": "True or False Does this have the message TEAM in it?"
        },
        "latefee": {
            "type": "number",
            "required": False,
            "description": "What is the dollar amount fee if a load is late? If not present return with 0",
        },
        "dropoff_time": {
            "type": "string",
            "required": False,
            "description": "What is the HH:MM dropoff time if it exists? If not present return with None"
        }
    }
}




def process_raw_text(model, tokenizer, json_schema, raw_text):
    #prompt_template = "{}"

    # TODO take the __init__ out of this method for speed (don't reload models)
    # though I could be wrong here
    jsonformer = JsonFormerText2Text(model, tokenizer, json_schema, raw_text, debug=False)
    generated_data = jsonformer()
    
    return generated_data



def main(force_json=True, verbose=False):

    if force_json:
        if model_type == 'text-generation':
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif model_type == 'text2text-generation':
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # TODO figure out load_in_8bit error later
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=False)

    loads, load_strs = make_random_loads(30)

    

    out = []
    elapsed = []
    for load_real, load_str in list(zip(loads, load_strs)):
        _start_time = time.time()

        if verbose:
            print(f"Input: {load_str}")
            print("LLM:")
        
        if force_json == True:
            pred = process_raw_text(model, tokenizer, json_schema, load_str)
        else:
            prompt = 'Your job is to extract information from the given message and populate a json sctructure using keywords from the message. The message: Flatbed load from Calverton, NY to Eagle, NE total weight 34271 lbs TEAM LOAD\nOutput result in the following JSON schema format:\n{"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "equipment": {"type": "string", "enum": ["Van", "Reefer", "Flatbed"], "description": "The type of truckload, often Van, Reefer, or Flatbed"}, "weight": {"type": "number"}, "weight_unit": {"type": "string"}, "hot": {"type": "boolean", "default": false, "description": "HOT load, hot"}, "team": {"type": "boolean", "default": false, "description": "If a load requires two drivers. Ex: TEAM, TEAM load"}, "latefee": {"type": "number", "required": false, "description": "dollar amount fee if a load is late"}, "dropoff time": {"type": "string", "required": false}}}\nResult: {"origin": '
            _pipeline = transformers.pipeline(
                model="databricks/dolly-v2-3b",
                torch_dtype=torch.float16,
                trust_remote_code=True, device_map="auto")
            pred = _pipeline([prompt])
    
        out.append(pred)

        elapsed.append(time.time() - _start_time)
        if verbose:
            print(pred)
            print(f"elapsed: {time.time() - _start_time}s")
            print("-"*40)
            print()

    # Compare results

    df_pred = pd.DataFrame(out)
    df_real = pd.DataFrame(loads)
    
    cols = df_pred.columns.str.capitalize()
    for col in cols:
        if col not in df_real.columns:
            df_real['Latefee'] = None
    df_real = df_real[cols]
    df_real = df_real.rename(columns=dict(
        zip(
            df_real.columns,
            df_real.columns.str.lower()
            )
        )
    )

    df_pred['origin_state'] = df_pred['origin_state'].str.upper()
    df_pred['destination_state'] = df_pred['destination_state'].str.upper()
    df_real['latefee'] = df_real['latefee'].fillna(0)
    df_real['hot'] = df_real['hot'].fillna(False)
    df_real['team'] = df_real['team'].fillna(False)


    df_agg = (df_real == df_pred)

    df_summary = df_agg.apply(pd.Series.value_counts)
    df_summary = df_summary.fillna(0)

    df_summary_percent = df_summary / len(df_agg)

    print(f"Model results for model: {model_name}")

    print(df_summary_percent.T)

    print(f"average time: {np.mean(elapsed)}")
    breakpoint()

if __name__ == '__main__':
    main(verbose=True)