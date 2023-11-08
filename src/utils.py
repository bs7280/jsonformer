import random
import pandas as pd
from termcolor import colored


df_cities = pd.read_csv("static/uscities.csv")

def make_random_load(json_schema, p_hot=0.7, p_team=0.5, p_dropoff=0.5, p_latefee=0.30):
    out = {}

    out['equipment'] = random.choice(['Van', 'Reefer', 'Flatbed'])
    out['weight'] = random.randint(0,45000)
    out['weight_unit'] = 'lbs'

    # Origin / Destination
    origin = df_cities.iloc[random.randint(0,df_cities.shape[0])]
    dest = df_cities.iloc[random.randint(0,df_cities.shape[0])]

    out['origin_city'] = f"{origin['city']}"
    out['origin_state'] = f"{origin['state_id']}"
    out['destination_city'] = f"{dest['city']}"
    out['destination_state'] = f"{dest['state_id']}"

    
    # Optional fields
    if random.random() < p_hot:
        out['hot'] = True #random.choice(['HOT', 'HOT LOAD'])
    else:
        out['hot'] = False
        
    if random.random() < p_team:
        out['team'] = True #random.choice(['TEAM', 'TEAM LOAD'])
    else:
        out['team'] = False
        
    if random.random() < p_dropoff:
        out['dropoff_time'] = f"{random.randint(6,23):02d}:00" 
    else:
        out['dropoff_time'] = json_schema.get('dropoff_time', {}).get('default') #False
        
    if random.random() < p_latefee:
        out['latefee'] = random.randint(50,250+1)
    else:
        out['latefee'] = json_schema.get('latefee', {}).get('default')

    return out

def make_random_loads(n_loads, target_schema):
    extra_cols = ['team', 'hot', 'dropoff_time', 'latefee']


    loads = [make_random_load(target_schema['properties']) for _ in range(n_loads)]
    load_strs = []

    for load in loads:
        details = []
        
        for ec in extra_cols:
            if ec in load and (load[ec] != False and load[ec] != target_schema['properties'].get(ec).get('default')):
                if ec == 'hot':
                    s = random.choice(['HOT', 'HOT LOAD'])
                elif ec == 'team':
                    s = random.choice(['TEAM', 'TEAM LOAD'])
                elif ec == 'dropoff_time':
                    s = f"{load[ec]}  Dropoff"
                elif ec == 'latefee':
                    s = f"${load[ec]} late fee"
                else:
                    s = load[ec]
                details.append(s)
        
        load_str = f"{load['equipment']} load from {load['origin_city']}, {load['origin_state']} to {load['destination_city']}, {load['destination_state']} total weight {load['weight']} lbs {' '.join(details)}"

        load_strs.append(load_str)

    return loads, load_strs

def highlight_values(value):
    def recursive_print(obj, indent=0, is_last_element=True):
        if isinstance(obj, dict):
            print("{")
            last_key = list(obj.keys())[-1]
            for key, value in obj.items():
                print(f"{' ' * (indent + 2)}{key}: ", end="")
                recursive_print(value, indent + 2, key == last_key)
            print(f"{' ' * indent}", end=",\n" if not is_last_element else "\n")
        elif isinstance(obj, list):
            print("[")
            for index, value in enumerate(obj):
                print(f"{' ' * (indent + 2)}", end="")
                recursive_print(value, indent + 2, index == len(obj) - 1)
            print(f"{' ' * indent}]", end=",\n" if not is_last_element else "\n")
        else:
            if isinstance(obj, str):
                obj = f'"{obj}"'
            print(colored(obj, "green"), end=",\n" if not is_last_element else "\n")

    recursive_print(value)
