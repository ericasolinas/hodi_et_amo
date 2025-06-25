import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df_pred = pd.read_csv("class_camoscio.csv")          
df_gold = pd.read_csv("HODI_2023_test_gold.tsv", sep="\t")   

df = df_pred.merge(df_gold[["id", "homotransphobic"]], on="id")

# Rimuovi righe dove la classificazione è 'errore'
df = df[df["Task A"].apply(lambda x: str(x).strip() in ["0", "1"])]
df["Task A"] = df["Task A"].astype(int)
df["homotransphobic"] = df["homotransphobic"].astype(int)

# Valutazione
print("\n=== Accuracy ===")
print(accuracy_score(df["homotransphobic"], df["Task A"]))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(df["homotransphobic"], df["Task A"]))

print("\n=== Classification Report ===")
print(classification_report(df["homotransphobic"], df["Task A"], digits=3))