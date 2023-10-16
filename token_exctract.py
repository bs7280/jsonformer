import transformers
import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

_pipeline = transformers.pipeline(model="facebook/bart-large-mnli")

out = _pipeline(
    ["Dry van picking up early morning only", "Chicago to Dallas, Truck order not used", "Reefer load"],
    candidate_labels=["Van", "Reefer", "Flatbed", "Morning", "Afternoon", "TONU"],
)

print(out)