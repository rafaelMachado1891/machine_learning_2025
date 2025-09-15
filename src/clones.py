import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import pyarrow

caminho= "../data/dados_clones.parquet"

df= pd.read_parquet(caminho)

# Index(['p2o_master_id', 'Massa(em kilos)', 'General Jedi encarregado',
#     'Estatura(cm)', 'Distância Ombro a ombro', 'Tamanho do crânio',
#       'Tamanho dos pés', 'Tempo de existência(em meses)', 'Status '],
#      dtype='object')
features = ['Distância Ombro a ombro', 'Estatura(cm)','Massa(em kilos)','Tamanho do crânio',
             'Tamanho dos pés', 'Tempo de existência(em meses)']

target = ['Status ']

y= df[target]
x= df[features]


x= x.replace({
    'Tipo 1': 1, 'Tipo 2': 2, 'Tipo 3': 3, 'Tipo 4': 4, 'Tipo 5': 5,
})

model= tree.DecisionTreeClassifier()

model.fit(X=x, y=y)

plt.figure(dpi=200)

tree.plot_tree(model,
          feature_names=features,
          class_names=model.classes_,
          filled=True,
          max_depth=3
          )

plt.show()

# pd.set_option("display.max_columns", None)

# print(x.head())


