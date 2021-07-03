import pandas as pd

print('Start!')
data = pd.DataFrame()

df_amazon = pd.read_csv('raw_files/amazon_cells_labelled.txt', header=None,sep='\t',names=['sentence','label'])
df_amazon['source'] = 'amazon'

df_imdb = pd.read_csv('raw_files/imdb_labelled.txt', header=None,sep='\t',names=['sentence','label'])
df_imdb['source'] = 'imdb'

df_yelp = pd.read_csv('raw_files/yelp_labelled.txt', header=None,sep='\t',names=['sentence','label'])
df_yelp['source'] = 'yelp'

data = data.append(df_amazon, ignore_index=True)
data = data.append(df_imdb, ignore_index=True)
data = data.append(df_yelp, ignore_index=True)

data.to_csv('dataset_merged.csv')

print('End')