from google.colab import drive
drive.mount('/content/drive')

#!pip install transformers accelerate bitsandbytes pandas

import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM
import pandas as pd
import time
from tqdm import tqdm

# Caricamento modello e tokenizer
model_name = "teelinsan/camoscio-7b-llama"
tokenizer = LlamaTokenizer.from_pretrained("linhvu/decapoda-research-llama-7b-hf")
model = LlamaForCausalLM.from_pretrained(
    "linhvu/decapoda-research-llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, "teelinsan/camoscio-7b-llama")

# Prompting

def generate_prompt(tweet):
    return f"""### Istruzione:

Sei un linguista specializzato in analisi di testi per l'identificazione di odio omotransfobico, ovvero odio verso la comunità LGBTQ+.

Ti verrà fornito un tweet. Il tuo compito è:
- Restituire solo **1** se contiene odio omotransfobico.
- Restituire solo **0** se non lo contiene.

Scrivi solo il numero, senza spiegazioni.


### Input:
{tweet}

### Risposta:"""

def genera_risposta(tweet):
    prompt = generate_prompt(tweet)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("### Risposta:")[-1].strip()

def parse_output(output):
    try:
        for line in output.splitlines():
            digits = ''.join(filter(str.isdigit, line))
            if digits in ("0", "1"):
                return int(digits)
        return "errore"
    except:
        return "errore"
    


# Caricamento del file CSV
df = pd.read_csv("/content/drive/MyDrive/HODI_2023_test_subtaskA.csv", sep="\t")

task_a = []

for tweet in tqdm(df["text"], desc="Classificazione tweet"):
    output = genera_risposta(tweet)
    label = parse_output(output)
    task_a.append(label)
    time.sleep(1.5)  

# Aggiunta colonna Task A
df["Task A"] = task_a

# Salvataggio del file finale
df[["id", "text", "Task A"]].to_csv("/content/drive/MyDrive/class_camoscio.csv", index=False)
print("\n File salvato in: /content/drive/MyDrive/class_camoscio.csv")