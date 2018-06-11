#%%
from gensim.models import Word2Vec

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

model_filename = 'gensim_W2Vmodel'


#%% 
sales = pd.read_pickle('sales_PKL.pkl')
print(sales.shape)
sales = sales[sales['Article'] != '49000002']
sales = sales[sales['Article'] != '49000003']
print(sales.shape)

#%%

bdd_rms = pd.read_excel('BDD.xlsx', names=['Article',
                                            'department',
                                            'model_adeo',
                                            'model_name',
                                            'date_created',
                                            'product_name',
                                            'is_stm'])

#%%

sales_merged = sales.copy()

sales_merged['Article'] = sales_merged['Article'].astype('str')
bdd_rms['Article'] = bdd_rms['Article'].astype('str')

sales_merged = sales_merged.merge(bdd_rms, on='Article')

print(sales_merged.head())

#%%

sales_categorical = pd.DataFrame(sales_merged['SaleDate'])
sales_categorical['CardNo'] = pd.DataFrame(sales_merged['CardNo'])
sales_categorical['model_adeo'] = pd.DataFrame(sales_merged['model_adeo'])
sales_categorical['PositionTotal'] = pd.DataFrame(sales_merged['PositionTotal'])
sales_categorical['ItemsSold'] = pd.DataFrame(sales_merged['ItemsSold'])

sales_categorical = sales_categorical.groupby(['SaleDate', 'CardNo','model_adeo'], as_index=False).agg('sum')


#%%

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

models_list = sales_categorical['model_adeo'].unique()



for model in models_list[0:1]:
    model_df = sales_categorical[sales_categorical['model_adeo'] == model].reset_index()
    items = model_df.shape[0]
    
    
    scale_df = pd.DataFrame(model_df['ItemsSold'])
    scale_df['PositionTotal'] = model_df['PositionTotal'] / model_df['ItemsSold']
    
    scaler = StandardScaler()
    scale_df = pd.DataFrame(scaler.fit_transform(scale_df), columns=scale_df.columns)
    
    kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(scale_df.values)
    
    model_df['ItemsScaled'] = scale_df.iloc[:, 0:1]
    model_df['PositionScaled'] = scale_df.iloc[:,1:2]
    model_df['Cluster'] = pd.DataFrame(y_kmeans)
    
    print(model_df.iloc[:, 4:9].head(50))
    
    items_std = model_df['ItemsSold'].std(axis=0)
    sum_std = model_df['PositionTotal'].std(axis=0)
    
    print('Модель {}, артикулов: {}, отклонение по штукам: {}, отклонение по сумме: {}'.format(model, items, items_std, sum_std))
    
    # Using the elbow method to find the optimal number of clusters
    
    

    
    
# Feature Scaling

    
    

#%%
import random

dev_set_list = []
test_set_list = []

clients_list = list(sales['CardNo'].unique())

len_cl = len(clients_list)
i = 0
for client in clients_list:
    i += 1
    if i % 1000 == 0:
        print('Обработано {} %'.format(int(i / len_cl * 100)))
    rn = random.random()
    if rn < 0.03:
        dev_set_list.append(sales[sales['CardNo'] == client])
    elif 0.03 <= rn < 0.06:
        test_set_list.append(sales[sales['CardNo'] == client])
    else:
        continue
    
dev_set = pd.concat(dev_set_list)
test_set = pd.concat(test_set_list)


#%%

dev_set.to_csv('dev_set.csv', sep='\t', encoding='utf-8')
test_set.to_csv('test_set.csv', sep='\t', encoding='utf-8')

#%%


clients_list = list(sales['CardNo'].unique())
dates_test = sales[sales['CardNo'] == clients_list[150000]]
dates_test.columns = dates_test.columns.str.replace('SaleDate', 'date')
dates_test.columns = dates_test.columns.str.replace('Article', 'product')
dates_test.columns = dates_test.columns.str.replace('CardNo', 'client')
dates_test.columns = dates_test.columns.str.replace('PositionTotal', 'sum')
from datetime import datetime



#%%
from datetime import datetime
def join_dates(df: pd.DataFrame):
    df = df.sort_values(by=['date'], ascending=[True])
    unique_dates = pd.DataFrame(df['date'].unique(), columns=['date']).astype(int)
    
    if len(unique_dates) == 1:
        return df
    
    unique_dates['new_date'] = unique_dates['date']
    
    dates_list = list(unique_dates['date'])
    for i in range(0, len(dates_list)-1, 2):
        if dates_list[i] == dates_list[i + 1] - 1:
            unique_dates.iat[i, 1] = dates_list[i + 1]
    
    new_dates_list = list(unique_dates['new_date'])
    for i in range(len(new_dates_list)-1, 0, -1):
        if new_dates_list[i] == new_dates_list[i - 1] + 1:
            unique_dates.iat[i, 1] = new_dates_list[i - 1]
    
    
    df = df.merge(unique_dates.astype(str), on='date')
    
    df['date'] = df['new_date']
    
    df = df.groupby(['date', 'product', 'client']).sum().reset_index()
    
    return df




#%% Разделить 
def split_client_df (df: pd.DataFrame):
    if len(df) == 1:
        return [df]
    
    df = df.sort_values(by=['date'], ascending=[True])
    
    # получим список уникальных дат
    dates_list = list(df['date'].unique())
    
    if len(dates_list) == 1:
        return 'pass'
    
    new_list = []
 
    
    for date in dates_list:
        
        new_df = df.copy()

        
        befor_df = new_df[new_df['date'] <= date].tail(15)
        after_df = new_df[new_df['date'] > date]
        
        if after_df.shape[0] == 0:
            continue

        
        after_df_during_month = after_df[after_df['date'] <= str(int(date) + 100)].copy()
        after_df_after_month = after_df[after_df['date'] > str(int(date) + 100)].copy()
        after_df_during_month['product'] = after_df_during_month['product'] + '+'
        after_df_after_month['product'] = after_df_after_month['product'] + '++'

        
        new_df = pd.concat([befor_df, after_df_during_month, after_df_after_month])
        
        new_list.append(new_df)


    return new_list

    
#%%

from multiprocessing.dummy import Pool as ThreadPool


def convert_to_sentences(data: pd.DataFrame,
                        client_column,
                        data_column,
                        product_column,
                        sales_column,
                        min_sum):

    sentences_list = []

    new_df = pd.DataFrame()
    new_df['date'] = data[data_column]
    new_df['client'] = data[client_column]
    new_df['product'] = data[product_column]
    new_df['sum'] = data[sales_column]

    new_df = new_df[new_df['sum'] >= min_sum]

    clients_list = list(new_df['client'].unique())

    print(len(clients_list))
    i = 0
    
    #pool = ThreadPool(8)
    
    #df_list = []
    
    #def get_df_list(client):

        #clients_df = new_df[new_df['client'] == client]
    for client in clients_list:
        
        i+=1
        
        if i % 100 == 0:
            print(i)
        
        clients_df = new_df[new_df['client'] == client]
        
        
        clients_df = join_dates(clients_df)
        
        if clients_df.shape[0] == 1:
            continue
        
        clients_df = clients_df.sort_values(by=['date', 'sum'], ascending=[True, False])
        
        #if clients_df.shape[0] == 1:
         #   return 1
        
        #clients_df = clients_df.sort_values(by=['date', 'sum'], ascending=[True, False])
        
        #split = split_client_df(clients_df)
    
        
        #for s in split:
         #   df_list.append(s)
         #   if len(df_list) % 100 == 0:
        #        print(len(df_list))

    #pool.map(get_df_list, clients_list)
    
    #pool.close()
    #pool.join()
    
    #print(len(df_list))
    
    
    #for s in df_list:
        products_list = list(clients_df['product'])
        products_list_str = list(map(str, products_list))
        sentences_list.append(products_list_str)
    

    return sentences_list

start = datetime.now()
codes_vect = convert_to_sentences(data=sales_categorical,
                            client_column='CardNo',
                            data_column='SaleDate',
                            product_column='model_adeo',
                            sales_column='PositionTotal',
                            min_sum=600)

print(datetime.now() - start)

#%%
import pickle

object_code_vec = codes_vect
file_code_vec = open('codes_vect.obj', 'wb')
pickle.dump(object_code_vec, file_code_vec)


#%%

df_tempo = pd.DataFrame([{'a': 0}, {'a': 1}])
df_tempo.to_csv(r'C:\Users\petr_\Dropbox\Future\file2.csv')

#%%
min_count = 15  # Встречается не реже чем такое количество раз
#size = 150  # Dimensionality of the feature vectors

# train word2vec on the sentences
model = Word2Vec(codes_vect, workers=4, window=7, sg=0, min_count=1, size=100, alpha=0.005, iter=200, compute_loss=True)
print(model.get_latest_training_loss() // 1000000)
model.save('w2v_33m_categorical')
model.save(r'C:\Users\petr_\Dropbox\Future\w2v_33m_categorical')

#%%

model.save('w2v_mymodel_33_min40_sg0_i250_window7_size130_transSplit')

#%%


unique_users = pd.DataFrame(sales_categorical['CardNo'].unique(), columns=['CardNo'])
unique_users['user_id'] = unique_users.index

unique_products = pd.DataFrame(sales_categorical['model_adeo'].unique(), columns=['model_adeo'])
unique_products['product_id'] = unique_products.index

unique_timesteps = pd.DataFrame(sales_categorical['SaleDate'].unique(), columns=['SaleDate'])
unique_timesteps['timestep_id'] = unique_timesteps.index

sales_categorical = sales_categorical.merge(unique_timesteps, on='SaleDate')


print(sales_categorical.columns)

#%%

sales_categorical['user_id'] = sales_categorical['user_id']+1
sales_categorical['product_id'] = sales_categorical['product_id']+1
sales_categorical['timestep_id'] = sales_categorical['timestep_id']+1

#%%

from spotlight.interactions import Interactions
from spotlight.sequence.implicit import ImplicitSequenceModel

implicit_interactions = Interactions(sales_categorical['user_id'].astype('int32').values, 
                                     sales_categorical['product_id'].astype('int32').values,
                                     timestamps=sales_categorical['timestep_id'].astype('int32').values)

sequential_interaction = implicit_interactions.to_sequence()

implicit_sequence_model = ImplicitSequenceModel()

#%%
start = datetime.now()
implicit_sequence_model = ImplicitSequenceModel(embedding_dim=100, representation='lstm', n_iter=5, use_cuda=True)
implicit_sequence_model.fit(sequential_interaction)
print(datetime.now() - start)

#%%

prediction = pd.DataFrame(implicit_sequence_model.predict([1337], item_ids=None), columns=['probability'])
prediction['product_id'] = prediction.index - 1
prediction = prediction.merge(unique_products, on='product_id')
prediction = prediction.merge(bdd_rms, on='model_adeo').drop_duplicates(['model_adeo'])
prediction['probability'] = prediction['probability'].astype('float')
prediction = prediction.sort_values(by=['probability'], ascending=[False])
print(prediction['model_name'].head(20))

#%%
def get_similar(lm_codes, topn):
    most_similar = model.wv.most_similar(positive=lm_codes, topn=topn)
    return most_similar


def predict_next_words(lm_codes, topn):
    predict_output_word = model.predict_output_word(lm_codes, topn=topn)
    return predict_output_word


similar = ['MOD_201189']
most_similar = get_similar(similar, topn=15)
print('Артикулы, наиболее похожие на {}'.format(similar))
for ms in most_similar:
    print(ms)

predict = ['MOD_201189']
print('После {} будет: '.format(predict))
predict_output_word = predict_next_words(predict, topn=15)
for pow in predict_output_word:
    print(pow)

loss = model.get_latest_training_loss()
