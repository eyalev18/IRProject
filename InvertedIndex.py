import pickle


with open(r'C:\Users\Owner\PycharmProjects\IRPtoject\IndexTry.pkl', 'rb') as inp:
    inverted = pickle.load(inp)
print(len(inverted.df))


#with open('IndexTry.pkl', 'wb') as outp:
    #pickle.dump(inverted, outp, pickle.HIGHEST_PROTOCOL)