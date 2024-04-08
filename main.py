from transformers import pipeline
import transformers
import accelerate
import time
from src.jsonformer import Jsonformer, JsonFormerText2Text, JsonFormerTrainDataGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import hashlib
from peft import PeftModel, PeftConfig

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import pandas as pd
import random
import torch
from src.utils import make_random_loads
import numpy as np
from datasets import Dataset

# Text-generation
model_name = 'tiiuae/falcon-7b'
model_name = 'databricks/dolly-v2-12b'
model_name = 'databricks/dolly-v2-3b'

## Text2text
model_type = 'text2text-generation'
model_name = 'google/flan-t5-large'
#model_name = 'google/flan-t5-xxl'
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
            "type": "number", 
            "description": "Get the weight as a number from this message"
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
            "default": 0,
            "description": "What is the dollar amount fee if a load is late? If not present return with 0",
        },
        "dropoff_time": {
            "type": "string",
            "default": '',
            "description": "What is the HH:MM dropoff time if it exists? If not present return with empty str"
        },
        "DOT_Numbers": {
            "type": "array",
            "items": {
                "DOT": {
                    "type": "string", 
                    'description': 'What is the DOT number of this message?',
                },
            }
        },
    }
}


json_schema_test = {
    "type": "object",
    "properties": {
        "DOT_Numbers": {
            "type": "array",
            "items": {
                "DOT": {
                    "type": "string", 
                    'description': 'What is the DOT number of this message?',
                },
            }
        },
    }
}

#json_schema['properties'].pop('DOT_number')




def process_raw_text(model, tokenizer, json_schema, raw_text):
    #prompt_template = "{}"

    # TODO take the __init__ out of this method for speed (don't reload models)
    # though I could be wrong here
    jsonformer = JsonFormerText2Text(model, tokenizer, json_schema, raw_text, debug=False)

    
    generated_data = jsonformer()
    
    return generated_data


def load_lora_model(peft_model_id):
    # Load LORA config (local file) which shouldn't be that big
    config = PeftConfig.from_pretrained(peft_model_id)

    # Load mdoels from 
    print(f"Loading base model from Peft lora config: {config.base_model_name_or_path}")
    model_name = config.base_model_name_or_path
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path) #, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_id)

    return model, tokenizer 

def load_base_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # NOTE 8 bit mode only works on GPU
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=False)
    return model, tokenizer

def make_training_data(n_loads):
    loads, load_strs = make_random_loads(n_loads, json_schema)

    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    # TODO figure out load_in_8bit error later
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=False)
    

    
    out = []
    elapsed = []
    for load_real, load_str in list(zip(loads, load_strs)):
        _start_time = time.time()
        jsonformer = JsonFormerTrainDataGenerator(None, None, json_schema, load_str, debug=False, real_value=load_real)
        generated_data = jsonformer()

        #pred = process_raw_text(model, tokenizer, json_schema, load_str)

        df_iter = pd.DataFrame({'real':pd.Series(load_real),'prompt':pd.Series(generated_data)})

        df_iter['input_hex'] = hashlib.md5(load_str.encode()).hexdigest()[:16]
        df_iter['raw_str'] = load_str
        out.append(df_iter)
        
    df_out = pd.concat(out)

    np.where(
        isinstance(
            df_out['real'].loc['hot'].iloc[0], bool
        ),
        df_out['real'].astype('str').str.lower(),
        df_out['real'].astype('str')
    )


    np.where(
        isinstance(df_out['real'], bool),
        df_out['real'].astype('str').str.lower(),
        df_out['real'].astype('str'))

    df_out['real'] = df_out['real'].astype('str').replace('True', 'true').replace('False', 'false')

    df_out['out_type'] = df_out.index.map(dict([(x[0],x[1].get('type')) for x in json_schema['properties'].items()]))

    ds = Dataset.from_dict({
        "prompt": df_out['prompt'],
        "real": df_out['real'],
        "out_type": df_out['out_type'],
        "out_col": df_out.index 
    })
    
    list_order = list(range(0, df_out.shape[0]))
    random.shuffle(list_order)

    df_out = df_out.iloc[list_order]

    ds.to_json(f'testdataset_{n_loads}.json')

def main(verbose=False, lora_path=None):


    if model_type == 'text-generation':
        raise ValueError("Stopped doing this")
        #model = AutoModelForCausalLM.from_pretrained(model_name)
        #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #elif model_type == 'text2text-generation':


    if lora_path:
        model, tokenizer = load_lora_model(peft_model_id=lora_path)
    else:
        model, tokenizer = load_base_model(model_name)



    loads, load_strs = make_random_loads(50, json_schema)

    

    out = []
    elapsed = []
    for load_real, load_str in list(zip(loads, load_strs)):
        _start_time = time.time()

        if verbose:
            print(f"Input: {load_str}")
            print("LLM:")
    
        pred = process_raw_text(model, tokenizer, json_schema_test, load_str)

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


    df_real = df_real[df_pred.columns]
    
    #cols = df_pred.columns.str.capitalize()
    #for col in cols:
    #    if col not in df_real.columns:
    #        df_real['Latefee'] = None
    #df_real = df_real[cols]
    #df_real = df_real.rename(columns=dict(
    #    zip(
    #        df_real.columns,
    #        df_real.columns.str.lower()
    #        )
    #    )
    #)

    #df_pred['origin_state'] = df_pred['origin_state'].str.upper()
    #df_pred['destination_state'] = df_pred['destination_state'].str.upper()
    #df_real['latefee'] = df_real['latefee'].fillna(0)
    #df_real['hot'] = df_real['hot'].fillna(False)
    #df_real['team'] = df_real['team'].fillna(False)


    df_agg = (df_real == df_pred)

    df_summary = df_agg.apply(pd.Series.value_counts)
    df_summary = df_summary.fillna(0)

    df_summary_percent = df_summary / len(df_agg)

    print(f"Model results for model: {model_name}")

    print(df_summary_percent.T)

    print(f"average time: {np.mean(elapsed)}")
    #breakpoint()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action")

    if False:
        loads, load_strs = make_random_loads(5, json_schema)

        print(load_strs)
        print()
        print()
        for l in loads:
            print(l)

    make_training_data(200)
    #main(verbose=True, lora_path=None)
    #main(verbose=True, lora_path="loras/lora_flan_l_1epoc_b/")
    #main(verbose=True, lora_path="loras/lora_flan_xxl_1epoc_a/")