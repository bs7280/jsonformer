import random
import pandas as pd
from termcolor import colored


df_cities = pd.read_csv("static/uscities.csv")

def make_random_load(p_hot=0.3, p_team=0.3, p_dropoff=0.8, p_latefee=0.33):
    out = {}

    out['Equipment'] = random.choice(['Van', 'Reefer', 'Flatbed'])
    out['Weight'] = random.randint(0,45000)
    out['Weight_unit'] = 'lbs'

    # Origin / Destination
    origin = df_cities.iloc[random.randint(0,df_cities.shape[0])]
    dest = df_cities.iloc[random.randint(0,df_cities.shape[0])]

    out['Origin_city'] = f"{origin['city']}"
    out['Origin_state'] = f"{origin['state_id']}"
    out['Destination_city'] = f"{dest['city']}"
    out['Destination_state'] = f"{dest['state_id']}"

    
    # Optional fields
    if random.random() < p_hot:
        out['Hot'] = True #random.choice(['HOT', 'HOT LOAD'])
        
    if random.random() < p_team:
        out['Team'] = True #random.choice(['TEAM', 'TEAM LOAD'])
        
    if random.random() < p_dropoff:
        out['Dropoff_time'] = f"{random.randint(6,23):02d}:00" 
        
    if random.random() < p_latefee:
        out['Latefee'] = random.randint(50,250+1)

    return out

def make_random_loads(n_loads):
    extra_cols = ['Team', 'Hot', 'Dropoff', 'Latefee']


    loads = [make_random_load() for _ in range(n_loads)]
    load_strs = []

    for load in loads:
        details = []
        
        for ec in extra_cols:
            if ec in load:
                if ec == 'Hot':
                    s = random.choice(['HOT', 'HOT LOAD'])
                elif ec == 'Team':
                    s = random.choice(['TEAM', 'TEAM LOAD'])
                elif ec == 'Dropoff_time':
                    s = f"{load[ec]}  Dropoff"
                elif ec == 'Latefee':
                    s = f"${load[ec]} late fee"
                else:
                    s = load[ec]
                details.append(s)
    
        
        load_str = f"{load['Equipment']} load from {load['Origin_city']}, {load['Origin_state']} to {load['Destination_city']}, {load['Destination_state']} total weight {load['Weight']} lbs {' '.join(details)}"

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
            print(f"{' ' * indent}}}", end=",\n" if not is_last_element else "\n")
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
