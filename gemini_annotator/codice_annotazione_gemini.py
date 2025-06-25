import google.generativeai as genai
import pandas as pd
import re 
import time
from google.genai import types  


#Variabili 
API_KEY = "AIzaSyAUbmlntw5A9yNDMUdG6gSR1c6r1mFCQw8"  
CSV_FILE = "HODI_2023_test_subtaskA.csv" 
OUTPUT_FILE = "annotated_data.csv" 
PARTIAL_SAVE_EVERY = 5 


#Inizializzazione API Gemini
client = genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")


#Creazione prompt
def create_prompt(text):
    prompt = f""" ###Istruzione:
Sei un linguista specializzato nell' identificazione di contenuti omotransfobici in tweet in lingua italiana. Il tuo compito è leggere ogni tweet e per ognuno PRIMA valuta se contiene odio omotransfobico o meno, POI classifica con:
[0]: se il tweet NON contiene odio omotransfobico
[1]: se il tweet CONTIENE odio omotransfobico.


Considera come odio omotransfobico qualsiasi espressione di pregiudizio, stereotipo,  discriminazione, disprezzo, derisione, incitamento all’odio o alla violenza verso persone LGBTQIA+ (esempio: appellativi quali 'finocchio', 'frocio', 'derivato', 'effeminato'). 
NON considerare omotransfobico l’uso di linguaggio volgare o offensivo che non sia a tema.


NON basarti solo su singole parole, ma valuta:
- il significato globale del tweet
- il tono usato (offensivo, ironico, denigratorio, neutro o positivo).


Fai attenzione a frasi ironiche, sarcastiche o ambigue: analizza il messaggio nel suo insieme.


Procedura da seguire:
1 - Assegnazione dell'etichetta binaria [0, 1]
2 - SOLO se assegni l’etichetta [1], formula una breve spiegazione nel formato : “Il tweet contiene odio omotransfobico perché [spiegazione]”; lascia vuoto in caso contrario.

Di seguito cinque esempi del formato che devi restituire [tweet, etichetta, spiegazione].
###Esempio 1:
Tweet: ve l’ho mai detto che i gay fanno schifo [URL]
Etichetta: 1
Spiegazione: Il tweet contiene odio omotransfobico perché l’espressione “i gay fanno schifo” è gravemente offensiva.
###Esempio 2:
Tweet: minuto 20 del primo episodio e mi è già partita a milla la ship tra leah e fatin. a volte odio essere così frocia
Etichetta: 0
Spiegazione:
###Esempio 3:
Tweet: Non hai abortito ed é nata una bambina rompicoglioni con una malattia non curabile + frocia non sei contenta mamma [URL]
Etichetta: 1
Spiegazione: Il tweet contiene odio omotransfobico perché l’appellativo ‘frocia’ viene preceduto da un ‘+’, che indica un’aggiunta peggiorativa: essere ‘frocia’ viene messo sullo stesso piano o addirittura in coda peggiorativa a condizioni già etichettate come “negative”.
###Esempio 4:
Tweet: I VERI BASTARDI VOTANO PER IL GAY PRIDE RUMENO!
Etichetta: 1
Spiegazione: Il tweet contiene odio omotransfobico perché la frase è omofoba perché associa in modo diretto il sostegno al Gay Pride a un insulto morale (“bastardi”), denigrando chi sostiene i diritti LGBTQIA+
###Esempio 5:
Tweet: i gay che non la votano sono omofobi [URL]
Etichetta: 0
Spiegazione:

Adesso, sulla base del formato sopra illustrato:
Tweet: {text}
Etichetta:
Spiegazione:"""

    return prompt   

#Funzione per interrogare l'API Gemini
def annotate_text(text, model):
    prompt = create_prompt(text)
    try:
        response = model.generate_content(prompt)
        output = response.text  

        label_match = re.search(r"Etichetta:\s*(\d)", output)
        label = label_match.group(1) if label_match else None

        explanation_match = re.search(r"Spiegazione:\s*(.*)", output, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        return label, explanation
        
    except Exception as e:
        print(f"Errore durante l'annotazione: {e}")
        return -1, "Annotazione fallita"  
    
#Caricamento dati dal CSV
def load_data(CSV_FILE):
    try:
        df = pd.read_csv(CSV_FILE, sep='\t', encoding='utf-8')
        return df
    except FileNotFoundError:
        print(f"Errore: File CSV '{CSV_FILE}' non trovato.")
        return None
    
#Annotazione dati
def annotate_data(df, model, OUTPUT_FILE):
    labels = []
    explanations = []

    for i, row in df.iterrows():
        test = row['text']
        label, explanation = annotate_text(test, model)
        labels.append(label)
        explanations.append(explanation)

        print(f"[{i+1}/{len(df)}] Annotato: {test} -> Etichetta: {label}, Spiegazione: {explanation}") 
        time.sleep(4)

        #Salvataggio parziale ogni PARTIAL_SAVE_EVERY iterazioni
        if (i + 1) % PARTIAL_SAVE_EVERY == 0:
            partial_df = df.iloc[:i+1].copy()
            partial_df['label'] = labels
            partial_df['explanation'] = explanations
            partial_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Salvataggio parziale dopo {i+1} annotazioni in '{OUTPUT_FILE}'")
   

    df['label'] = labels
    df['explanation'] = explanations
    return df

#Salvataggio risultati 
def save_results(df, output_file):
    try:
        df.to_csv(output_file, index=False)
        print(f"Risultati salvati in '{output_file}'")
    except Exception as e:
        print(f"Errore durante il salvataggio: {e}")

#Funzione main
def main():
    df = load_data(CSV_FILE)
    if df is None:
        return  

    annotated_df = annotate_data(df, model, OUTPUT_FILE)

    if annotated_df is not None:  
        save_results(annotated_df, OUTPUT_FILE)


#Esecuzione del codice
if __name__ == "__main__":
    main()