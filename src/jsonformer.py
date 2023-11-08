from typing import List, Union, Dict, Any

from src.logits_processors import (
    NumberStoppingCriteria,
    OutputNumbersTokens,
    StringStoppingCriteria,
)
from termcolor import cprint
from transformers import PreTrainedModel, PreTrainedTokenizer
import json

GENERATION_MARKER = "|GENERATION|"


class Jsonformer:
    value: Dict[str, Any] = {}

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        json_schema: Dict[str, Any],
        prompt: str,
        *,
        debug: bool = False,
        max_array_length: int = 10,
        max_number_tokens: int = 6,
        temperature: float = 1.0,
        max_string_token_length: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.json_schema = json_schema
        self.prompt = prompt

        self.number_logit_processor = OutputNumbersTokens(self.tokenizer, self.prompt)

        self.generation_marker = "|GENERATION|"
        self.debug_on = debug
        self.max_array_length = max_array_length

        self.max_number_tokens = max_number_tokens
        self.temperature = temperature
        self.max_string_token_length = max_string_token_length

    def debug(self, caller: str, value: str, is_prompt: bool = False):
        if self.debug_on:
            if is_prompt:
                cprint(caller, "green", end=" ")
                cprint(value, "yellow")
            else:
                cprint(caller, "green", end=" ")
                cprint(value, "blue")

    def get_prompt_length(self, input_tokens):
        return len(input_tokens)

    def generate_number(self, temperature: Union[float, None] = None, iterations=0):
        prompt = self.get_prompt()

        self.debug("[generate_number]", prompt, is_prompt=True)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        #breakpoint()
        response = self.model.generate(
            input_ids=input_tokens,
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            logits_processor=[self.number_logit_processor],
            stopping_criteria=[
                NumberStoppingCriteria(self.tokenizer, self.get_prompt_length(input_tokens[0]))
            ],
            temperature=temperature or self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(response[0], skip_special_tokens=True)
        print(response)
        breakpoint()

        response = response[self.get_prompt_length(prompt) :]
        response = response.strip().rstrip(".")
        self.debug("[generate_number]", response)
        try:
            return float(response)
        except ValueError:
            if iterations >= 0:
                raise ValueError(f"Failed to generate a valid number response: {response}")

            return self.generate_number(temperature=self.temperature * 1.3, iterations=iterations + 1)

    def generate_boolean(self) -> bool:
        prompt = self.get_prompt()
        self.debug("[generate_boolean]", prompt, is_prompt=True)

        # Note - ive noticed without the + pad_token, model gets very confused
        # since it by default wants to start with a prediction of [0, <true/false token id>, 1]
        # where 0 is pad token, 1 is EOS token
        input_tensor = self.tokenizer.encode(prompt + self.tokenizer.pad_token, return_tensors="pt")
        output = self.model.forward(input_tensor.to(self.model.device))
        logits = output.logits[0, -1]

        # todo: this assumes that "true" and "false" are both tokenized to a single token
        # this is probably not true for all tokenizers
        # this can be fixed by looking at only the first token of both "true" and "false"
        true_token_id = self.tokenizer.convert_tokens_to_ids("true")
        false_token_id = self.tokenizer.convert_tokens_to_ids("false")

        result = logits[true_token_id] > logits[false_token_id]

        self.debug("[generate_boolean]", result)

        return result.item()

    def generate_string(self) -> str:
        prompt = self.get_prompt() + '"'
        self.debug("[generate_string]", prompt, is_prompt=True)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )

        response = self.model.generate(
            input_ids=input_tokens,
            max_new_tokens=self.max_string_token_length,
            num_return_sequences=1,
            temperature=self.temperature,
            stopping_criteria=[
                StringStoppingCriteria(self.tokenizer, self.get_prompt_length(input_tokens[0]))
            ],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Some models output the prompt as part of the response
        # This removes the prompt from the response if it is present
        if (
            len(response[0]) >= len(input_tokens[0])
            and (response[0][: len(input_tokens[0])] == input_tokens).all()
        ):
            response = response[0][len(input_tokens[0]) :]
        if response.shape[0] == 1:
            response = response[0]

        response = self.tokenizer.decode(response, skip_special_tokens=True)

        self.debug("[generate_string]", "|" + response + "|")

        if response.count('"') < 1:
            return response

        return response.split('"')[0].strip()

    def generate_object(
        self, properties: Dict[str, Any], obj: Dict[str, Any]
    ) -> Dict[str, Any]:
        for key, schema in properties.items():
            self.debug("[generate_object] generating value for", key)
            obj[key] = self.generate_value(schema, obj, key)
        return obj

    def generate_value(
        self,
        schema: Dict[str, Any],
        obj: Union[Dict[str, Any], List[Any]],
        key: Union[str, None] = None,
    ) -> Any:
        schema_type = schema["type"]
        if schema_type == "number":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_number()
        elif schema_type == "boolean":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_boolean()
        elif schema_type == "string":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_string()
        elif schema_type == "array":
            new_array = []
            obj[key] = new_array
            return self.generate_array(schema["items"], new_array)
        elif schema_type == "object":
            new_obj = {}
            if key:
                obj[key] = new_obj
            else:
                obj.append(new_obj)
            return self.generate_object(schema["properties"], new_obj)
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def generate_array(self, item_schema: Dict[str, Any], obj: Dict[str, Any]) -> list:
        for _ in range(self.max_array_length):
            # forces array to have at least one element
            element = self.generate_value(item_schema, obj)
            print(f"Respone: {element}")
            obj[-1] = element

            obj.append(self.generation_marker)
            input_prompt = self.get_prompt()
            obj.pop()
            input_tensor = self.tokenizer.encode(input_prompt, return_tensors="pt")
            output = self.model.forward(input_tensor.to(self.model.device))
            logits = output.logits[0, -1]


            top_indices = logits.topk(30).indices
            sorted_token_ids = top_indices[logits[top_indices].argsort(descending=True)]

            found_comma = False
            found_close_bracket = False

            for token_id in sorted_token_ids:
                decoded_token = self.tokenizer.decode(token_id)
                if ',' in decoded_token:
                    found_comma = True
                    break
                if ']' in decoded_token:
                    found_close_bracket = True
                    break

            if found_close_bracket or not found_comma:
                break

        return obj

    def get_prompt(self, next_key=None, next_obj=None):
        # Find key of next token
        if next_key is None:
            next_key = list(self.value.keys())[-1]
        if next_obj is None:
            next_obj = self.json_schema['properties'].get(next_key)

        template = """{prompt}\nOutput result in the following JSON schema format:\n{schema}\nDescription for next item:\n{description}\nResult: {progress}"""

        progress = json.dumps(self.value)
        gen_marker_index = progress.find(f'"{self.generation_marker}"')
        if gen_marker_index != -1:
            progress = progress[:gen_marker_index]
        else:
            raise ValueError("Failed to find generation marker")

        desc = next_obj.get('description', '')

        prompt = template.format(
            prompt=self.prompt,
            schema=json.dumps(self.json_schema),
            description=desc,
            progress=progress,
        )

        return prompt

    def __call__(self) -> Dict[str, Any]:
        self.value = {}
        generated_data = self.generate_object(
            self.json_schema["properties"], self.value
        )
        return generated_data


class JsonFormerText2Text(Jsonformer):


    def get_prompt_length(self, prompt):
        return 0

    def generate_number(self, temperature: Union[float, None] = None, iterations=0):
        prompt = self.get_prompt()

        self.debug("[generate_number]", prompt, is_prompt=True)
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        #breakpoint()
        response = self.model.generate(
            input_ids=input_tokens,
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            logits_processor=[self.number_logit_processor],
            stopping_criteria=[
                NumberStoppingCriteria(self.tokenizer, 0) # No input tokens buffer with text2text
            ],
            temperature=temperature or self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        response = self.tokenizer.decode(response[0], skip_special_tokens=True)

        # text2text does not include prompt in output
        # response = response[len(prompt) :]

        # Theres a weird bug in FastTransformers that causes a space to be added before every token??
        response = response.replace(' ', '')


        response = response.strip().rstrip(".")
        self.debug("[generate_number]", response)
        try:
            return float(response)
        except ValueError:
            if iterations >= 3:
                print(f"Warning Failed to generate a valid number response: {response}")
                #warn ValueError(f"Failed to generate a valid number response: {response}")
                return None

            return self.generate_number(temperature=self.temperature * 1.3, iterations=iterations + 1)


    def generate_boolean(self) -> bool:
        prompt = self.get_prompt()
        self.debug("[generate_boolean]", prompt, is_prompt=True)

        # Note - ive noticed without the + pad_token, model gets very confused
        # since it by default wants to start with a prediction of [0, <true/false token id>, 1]
        # where 0 is pad token, 1 is EOS token
        input_tensor = self.tokenizer.encode(prompt + self.tokenizer.pad_token, return_tensors="pt")
        #input_tensor = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_tensor.to(self.model.device)
        output = self.model.forward(input_ids, decoder_input_ids=input_ids)
        logits = output.logits[0, -1]

        # todo: this assumes that "true" and "false" are both tokenized to a single token
        # this is probably not true for all tokenizers
        # this can be fixed by looking at only the first token of both "true" and "false"
        true_token_id = self.tokenizer.encode('true')[0] #self.tokenizer.convert_tokens_to_ids("yes") #"True")
        false_token_id = self.tokenizer.encode('false')[0] #self.tokenizer.convert_tokens_to_ids("no") #"False")

        # TODO post run - use:

        #breakpoint()


        # (Pdb) self.tokenizer.encode('False')[0]
        # 10747
        # (Pdb) self.tokenizer.encode('True')[0]
        # 10998
        # (Pdb) logits[10998]
        # tensor(-19.4769, grad_fn=<SelectBackward0>)
        # (Pdb) logits[10747]
        # tensor(-22.0907, grad_fn=<SelectBackward0>)

        if true_token_id == false_token_id:
            raise ValueError("Error, true and false token ids are identical for this model.")

        result = logits[true_token_id] > logits[false_token_id]

        self.debug("[generate_boolean]", result)

        return result.item()

    def get_prompt(self, next_key=None, next_obj=None):
        # Find key of next token
        if next_key is None:
            next_key = list(self.value.keys())[-1]
        if next_obj is None:
            next_obj = self.json_schema['properties'].get(next_key)



        desc = next_obj.get('description', None)
        d_type = next_obj.get('type')

        if desc:
            header = f"{desc}"
        else:
            header = f"What is the {next_key} from the following message as a {d_type}?"

       #template = """{prompt}\nOutput result in the following JSON schema format:\n{schema}\nDescription for next item:\n{description}\nResult: {progress}"""

        prompt = f"""{header} \n Message: {self.prompt}"""


        #breakpoint()

        if False:
            progress = json.dumps(self.value)
            gen_marker_index = progress.find(f'"{self.generation_marker}"')
            if gen_marker_index != -1:
                progress = progress[:gen_marker_index]
            else:
                raise ValueError("Failed to find generation marker")

            if False:
                prompt = template.format(
                    prompt=self.prompt,
                    schema=json.dumps(self.json_schema),
                    description=desc,
                    progress=progress,
                )
            
        return prompt

        
class JsonFormerTrainDataGenerator(JsonFormerText2Text):

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        json_schema: Dict[str, Any],
        prompt: str,
        *,
        debug: bool = False,
        max_array_length: int = 10,
        max_number_tokens: int = 6,
        temperature: float = 1.0,
        max_string_token_length: int = 10,
    ):
        #self.model = model
        #self.tokenizer = tokenizer
        self.json_schema = json_schema
        self.prompt = prompt

        #self.number_logit_processor = OutputNumbersTokens(self.tokenizer, self.prompt)

        self.generation_marker = "|GENERATION|"
        self.debug_on = debug
        self.max_array_length = max_array_length

        self.max_number_tokens = max_number_tokens
        self.temperature = temperature
        self.max_string_token_length = max_string_token_length


    def generate_value(
        self,
        schema: Dict[str, Any],
        obj: Union[Dict[str, Any], List[Any]],
        key: Union[str, None] = None,
    ) -> Any:
        schema_type = schema["type"]



        if schema_type in ("number", "string", "boolean"):
            prompt = self.get_prompt(next_key=key)
            return prompt
        else:
            print(schema_type)
            breakpoint()


        if schema_type == "number":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_number()
        elif schema_type == "boolean":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_boolean()
        elif schema_type == "string":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_string()
        elif schema_type == "array":
            new_array = []
            obj[key] = new_array
            return self.generate_array(schema["items"], new_array)
        elif schema_type == "object":
            new_obj = {}
            if key:
                obj[key] = new_obj
            else:
                obj.append(new_obj)
            return self.generate_object(schema["properties"], new_obj)
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")
