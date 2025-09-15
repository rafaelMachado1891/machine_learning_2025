import pandas as pd
import openpyxl
from sklearn import tree
import matplotlib.pyplot as plt

caminho = "../data/dados_cerveja.xlsx"

df= pd.read_excel(caminho)

features = ['copo','espuma','cor', 'temperatura']
target = 'classe'

x= df[features]
y= df[target]

x= x.replace({
    "mud": 0, "pint": 1,
    "não": 0, "sim": 1,
    "escura": 0, "clara": 1
})

model = tree.DecisionTreeClassifier()

model.fit(X=x,y=y)

plt.figure(dpi=200)

tree.plot_tree(model, feature_names=features,
               class_names=model.classes_,
               filled=True
               )

plt.show()

