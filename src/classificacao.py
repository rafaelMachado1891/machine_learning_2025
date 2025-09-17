import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from sklearn import linear_model

df = pd.read_excel("../data/dados_cerveja_nota.xlsx")

df['aprovado'] = (df['nota'] >= 5).astype(int)

plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title('cerveja x aprovacao')
plt.xlabel('cerveja')
plt.ylabel('nota')

plt.show()

reg = linear_model.LogisticRegression(penalty=None,
                                fit_intercept=True)

reg.fit(df['cerveja'], df['aprovado'])
reg_predict = reg.predict(df[['cerveja']].drop_duplicates())

print(reg_predict)