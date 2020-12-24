import pickle

model = pickle.load(open('Pickle_ML_File.pkl', 'rb'))

print(model.support_)
print(model.ranking_)

#print(model)