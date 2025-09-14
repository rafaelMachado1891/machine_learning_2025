# %%
import pandas as pd
import openpyxl
from sklearn import tree
import matplotlib.pyplot as plt

# %%
caminho = '../data/dados_frutas.xlsx'

df=pd.read_excel(caminho)

print(df)

arvore = tree.DecisionTreeClassifier(random_state=42)

y= df["Fruta"]

caracteristicas = ["Arredondada",  "Suculenta",  "Vermelha", "Doce"]

X= df[caracteristicas]

arvore.fit(X,y)

#arvore.predict([[0,0,0,0]])

plt.figure(dpi=400, figsize=[4,4])

tree.plot_tree(arvore,
               feature_names=caracteristicas,
               class_names=arvore.classes_,
               filled=True
               )

proba = arvore.predict_proba([[1,1,1,0]])[0]

print(pd.Series(proba, index=arvore.classes_))