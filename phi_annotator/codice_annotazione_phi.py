from google.colab import drive
drive.mount('/content/drive')

#Installazione librerie
!pip install huggingface_hub
!pip install -U bitsandbytes #funzionalità che permette il caricamento dei modelli in modalità 4-bit e riduce l'uso di memoria
!pip install transformers accelerate 

#Importazione moduli 
import os
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.phi3.modeling_phi3 import DynamicCache  #correzione necessaria per i modelli Phi - 3.5

#FIX per modelli Phi-3.5
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = DynamicCache.get_seq_length

#Variabili 
model_id = "anakin87/Phi-3.5-mini-ITA"
PARTIAL_SAVE_EVERY = 5
text_column = "text"
output_path = "dati_classificati_phi3.csv"

#Caricamento modello e tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)

#Funzione di classificazione 
def classify_and_explain(text, max_retries=2):
    prompt = f"""
### Istruzione:

Sei un linguista specializzato in analisi di testi per l'identificazione di odio omotransfobico, ovvero odio verso la comunità LGBTQ+.

Ti verrà fornito un tweet. Il tuo compito è:
• Restituire solo *1* se contiene odio omotransfobico.
• Restituire solo *0* se non lo contiene.

Scrivi solo il numero, senza spiegazioni.

### Tweet:
{text}

### Etichetta:
"""

    for attempt in range(max_retries):
        try:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            generated_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                use_cache=False,
                eos_token_id=tokenizer.eos_token_id
            )

            decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated = decoded_output[len(prompt):].strip()

            match = re.search(r"(\d)", generated)
            if match:
                label = int(match.group(1))
                return label, ""  

        except Exception as e:
            print(f"Tentativo {attempt+1} fallito: {e}")

    return -1, ""

#Caricamento dati 
df = pd.read_csv("/content/drive/MyDrive/HODI_2023_test_subtaskA.csv", sep="\t")

labels = []
explanations = []


for i, text in enumerate(tqdm(df[text_column].fillna(""))): #fillna("") per gestire eventuali valori NaN
    label, explanation = classify_and_explain(text)
    labels.append(label)
    explanations.append(explanation)

#Salvataggio parziale 
    if (i + 1) % PARTIAL_SAVE_EVERY == 0:
        partial_df = df.iloc[:i+1].copy()
        partial_df['predizione'] = labels
        partial_df['spiegazione'] = explanations
        partial_df.to_csv(output_path, index=False)
        print(f"Salvataggio parziale dopo {i+1} annotazioni in '{output_path}'")

#Salvataggio finale 
df['predizione'] = labels
df['spiegazione'] = explanations

df.to_csv(output_path, index=False)
print(f" Classificazione completata e file salvato in: {output_path}")
