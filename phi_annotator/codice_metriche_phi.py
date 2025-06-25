import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Caricamento file da confrontare
df_pred = pd.read_csv("/content/dati_classificati_phi3.csv")
df_gold = pd.read_csv("/content/HODI_2023_test_GOLD.csv", sep="\t", encoding='latin-1', engine='python')

#Merge
df = df_pred.merge(df_gold[["id", "homotransphobic"]], on="id")

#Rimozione righe con NaN
df = df.dropna(subset=["predizione", "homotransphobic"])

#Conversione a interi
y_true = df["homotransphobic"].astype(int)
y_pred = df["predizione"].astype(int)

#Rimozione predizioni non valide 
df = df[df["predizione"].isin([0, 1])]

#Valutazione
print("\n=== Accuracy ===")
print(accuracy_score(y_true, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, digits=3))