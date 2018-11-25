import dataset

f = dataset.Dataset("data/Dados-medicos.csv")

print float(f.data.iloc[0]["IDADE"])

f.corr()
