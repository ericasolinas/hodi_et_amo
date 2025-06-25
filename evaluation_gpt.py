import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df_pred = pd.read_csv("class_gpt.csv")          
df_gold = pd.read_csv("HODI_2023_test_gold.tsv", sep="\t")   

df = df_pred.merge(df_gold[["id", "homotransphobic"]], on="id")

y_true = df["homotransphobic"].astype(int)
y_pred = df["Task A"].astype(int)

# Valutazione
print("\n=== Accuracy ===")
print(accuracy_score(y_true, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, digits=3))