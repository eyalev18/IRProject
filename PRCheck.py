# import pandas as pd
#
# df = pd.read_csv('PageRank.csv',header = None)
#
# #print(df)
#
# df.columns=['0','1']
# for col in df.columns:
#     print(col)
# #
# lst=[3434750,10568,32927]
# res=[]
# for i in lst:
#     res.append(df.loc[df['0'] == i, '1'].iloc[0])
# print(res)



lst=[3434750,10568,32927]
res=[]
for i in lst:
    res.append(counter[i])
print(res)
