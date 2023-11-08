import openai
import os
import pandas as pd
import time
import json
import numpy as np

from main import json_schema
from src.utils import make_random_loads

def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message["content"]

openai.api_key = os.environ.get('OPENAI_KEY')

load_real, load_str = make_random_loads(1, json_schema)

def get_prompt(load):
    prompt = f"""Return a valid JSON matching this schema for the message below

    JsonSchema: 
    {json_schema}

    Message:
    {load}
    """

    return prompt

loads, load_strs = make_random_loads(50, json_schema)

out = []
elapsed = []
for load_real, load_str in list(zip(loads, load_strs)):
    _start_time = time.time()

    prompt = get_prompt(load_str)
    response = get_completion(prompt)

    out.append(response)

    print(load_str)
    print(response)
    print()

    elapsed.append(time.time() - _start_time)


out = [json.loads(x) for x in out]

# Compare results
df_pred = pd.DataFrame(out)
df_real = pd.DataFrame(loads)


df_real = df_real[df_pred.columns]


df_agg = (df_real == df_pred)

df_summary = df_agg.apply(pd.Series.value_counts)
df_summary = df_summary.fillna(0)

df_summary_percent = df_summary / len(df_agg)

print(f"Model results for model: gpt-3.5-turbo")

print(df_summary_percent.T)

print(f"average time: {np.mean(elapsed)}")

breakpoint()


