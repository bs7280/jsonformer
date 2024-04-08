import random
import pandas as pd
from termcolor import colored
import numpy as np
import string


df_cities = pd.read_csv("static/uscities.csv")

# Add MC numbers
def get_random_letter_number(length):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

def make_random_load(json_schema, p_hot=0.7, p_team=0.5, p_dropoff=0.9, p_latefee=0.30):
    out = {}

    out['equipment'] = random.choice(['Van', 'Reefer', 'Flatbed'])
    # Weight
    out['weight'] = random.randint(0, 45000)

    #if random.choice() > 0.7:
    #    out['weight'] =  "{:,}".format(out['weight'])
    #random_selection = np.random.choice(a=[False, True], size=(1, 10), p=[0.5, 1-0.5])
    #np.where(random_selection, out['weight'], out['weight'].apply(lambda x:))
    
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

    if random.random() <= 0.9999:
            # first range is length of fake id, second is for number of items
            fake_ids = [get_random_letter_number(random.randint(6,20)) for l in range(random.randint(1,5))]
            dot_numbers = ', '.join([f"DOT: {x}" for x in fake_ids])
            out['DOT_Numbers'] = dot_numbers
            #load['DOT_Numbers'] = fake_ids
            #dot_numbers = ' '.join([f"DOT: {x}" for x in fake_ids])
            #dot_numbers = "DOT Numbers " + dot_numbers

    return out

def make_random_loads(n_loads, target_schema):
    extra_cols = ['team', 'hot', 'dropoff_time', 'latefee']


    loads = [make_random_load(target_schema['properties']) for _ in range(n_loads)]
    load_strs = []

    # Change weight to occasionally have commas
    #random_selection = np.random.choice(a=[False, True], size=(1, 10), p=[0.5, 1-0.5])
    #details['weight'] = np.where(
    #    random_selection, details['weight'].values(),
    #    ["{:,}".format(x) for x in details['weight']]
    #    )
    
    for load in loads:
        details = []
        
        load_out = load.copy()
        for ec in extra_cols:
            if ec in load and (load[ec] != False and load[ec] != target_schema['properties'].get(ec).get('default')):
                if ec == 'hot':
                    s = random.choice(['HOT', 'HOT LOAD'])
                elif ec == 'team':
                    s = random.choice(['TEAM', 'TEAM LOAD'])
                elif ec == 'dropoff_time':
                    # Get HH:MM or HH AM/PM
                    r_timeformat = random.random()
                    if r_timeformat < 0.4:
                        t = int(load[ec].split(":")[0])
                        if t < 12:
                            s_time = f"{t} AM"
                        else:
                            s_time = f"{t} PM"
                    else:
                        s_time = f"{load[ec]}"

                    # Get Delivers / Dropoff + <time>
                    r_deliverformat = random.random()
                    if r_deliverformat < 0.39:
                        s_dropoff = "dropoff"
                    elif r_deliverformat < 0.69:
                        s_dropoff = "delivery"
                    else:
                        s_dropoff = "delivers"

                    if random.random() <= 0.5:
                        s = f"{s_dropoff} {s_time}"
                    else:
                        s = f"{s_time} {s_dropoff}"
                elif ec == 'latefee':
                    s = f"${load[ec]} late fee"
                else:
                    s = load[ec]
                details.append(s)
        

        if random.random() > 0.5:
            load_out['weight'] =  "{:,}".format(load['weight'])

            if random.random() > 0.7:
                load_out['weight'] = "{}.00".format(load['weight'])
        


        if 'DOT_Numbers' in load:
            # first range is length of fake id, second is for number of items
            #fake_ids = [get_random_letter_number(random.randint(6,20)) for l in range(random.randint(1,5))]
            #load['DOT_Numbers'] = fake_ids
            #dot_numbers = ' '.join([f"DOT: {x}" for x in fake_ids])
            dot_numbers = "DOT Numbers " + load['DOT_Numbers']
        else:
            dot_numbers = ''


        load_str = (
            f"{load_out['equipment']} load from {load_out['origin_city']}, {load_out['origin_state']} to "
            f"{load_out['destination_city']}, {load_out['destination_state']} total weight {load_out['weight']} "
            f"lbs {' '.join(details)} {dot_numbers}")

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
