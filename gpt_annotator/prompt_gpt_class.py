import openai
import pandas as pd
import time

# Configurazione modello ed API
openai.api_key = "API_KEY"

MODEL = "gpt-4.1-mini"  
CSV_PATH = "/Users/ericasolinas/Desktop/UNIBO/MNNLP/PROGETTO_HODI/HODI_2023_test/HODI_2023_test_subtaskA.csv"  
OUTPUT_PATH = "class_gpt.csv"
COLUMN_NAME = "text"

# Prompting
system_prompt = (
    	"""
Sei un linguista specializzato nell'identificazione di contenuti omotransfobici in tweet in lingua italiana.
Il tuo compito è leggere ogni tweet e per ognuno PRIMA valutare se contiene odio omotransfobico o meno,	POI classificarlo con:
[0]: se il tweet NON contiene odio omotransfobico
[1]: se il tweet CONTIENE odio omotransfobico.
Considera come odio omotransfobico qualsiasi espressione di pregiudizio, stereotipo, discriminazione, disprezzo, derisione, incitamento all’odio o alla violenza verso persone LGBTQIA+.
NON considerare omotransfobico l’uso di linguaggio volgare o offensivo che non sia a tema.
NON basarti solo su singole parole, ma valuta:
- il significato globale del tweet
- il tono usato (offensivo, ironico, denigratorio, neutro o positivo).
Fai attenzione a frasi ironiche, sarcastiche o ambigue: analizza il messaggio nel suo insieme.
)
"""

)

few_shot_examples = """Esempio 1:
Tweet: ve l’ho mai detto che i gay fanno schifo [URL]
Etichetta: 1
Spiegazione: Il tweet contiene odio omotransfobico perché l’espressione “i gay fanno schifo” è gravemente offensiva.
Esempio 2:
Tweet: minuto 20 del primo episodio e mi è già partita a milla la ship tra leah e fatin. a volte odio essere così frocia
Etichetta: 0
Spiegazione:
Esempio 3:
Tweet: Non hai abortito ed é nata una bambina rompicoglioni con una malattia non curabile + frocia non sei contenta mamma [URL]
Etichetta: 1
Spiegazione: Il tweet contiene odio omotransfobico perché l’appellativo ‘frocia’ viene preceduto da un ‘+’, che indica un’aggiunta peggiorativa.
Esempio 4:
Tweet: I VERI BASTARDI VOTANO PER IL GAY PRIDE RUMENO!
Etichetta: 1
Spiegazione: Il tweet contiene odio omotransfobico perché associa in modo diretto il sostegno al Gay Pride a un insulto morale (“bastardi”).
Esempio 5:
Tweet: i gay che non la votano sono omofobi [URL]
Etichetta: 0
Spiegazione:
"""

# Classificazione
def analizza_tweet(tweet):
    user_prompt = (
        few_shot_examples
        + f"Adesso, sulla base del formato sopra illustrato:\nTweet: {tweet}\nEtichetta:"
    )

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=200,
    )

    output = response["choices"][0]["message"]["content"].strip()

        # Parsing output
    try:
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        etichetta = int(lines[0].replace("Etichetta:", "").strip("[] ").strip())
        spiegazione = lines[1].replace("Spiegazione:", "").strip("[] ").strip() if etichetta == 1 and len(lines) > 1 else ""
        return etichetta, spiegazione
    except:
        return "errore", output



df = pd.read_csv(CSV_PATH, encoding = "utf-8", sep = "\t") 
task_a = []
task_b = []

for i, tweet in enumerate(df[COLUMN_NAME]):
    print(f"\nAnalizzando tweet {i+1}/{len(df)}...")
    etichetta, spiegazione = analizza_tweet(tweet)
    task_a.append(etichetta)
    task_b.append(spiegazione)
    time.sleep(1.5)  

# Aggiunta colonne Task A e B e salvataggio
df["Task A"] = task_a
df["Task B"] = task_b
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nClassificazione completata. File salvato in: {OUTPUT_PATH}")