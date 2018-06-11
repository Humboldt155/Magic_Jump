#%%
from gensim.models import Word2Vec

import pandas as pd
import numpy as np

model_filename = 'gensim_W2Vmodel'

from datetime import timedelta


#%% 
sales = pd.read_pickle('preprocessing/sales_13_months_0.pkl')
print(sales.shape)

#%%


clients_list = list(sales['CardNo'].unique())
# dates_test = sales[sales['CardNo'] == clients_list[45003]]
dates_test = sales.head(1000)
dates_test.columns = dates_test.columns.str.replace('SaleDate', 'date')
dates_test.columns = dates_test.columns.str.replace('Article', 'product')
dates_test.columns = dates_test.columns.str.replace('CardNo', 'client')
dates_test.columns = dates_test.columns.str.replace('PositionTotal', 'sum')



#%%

from datetime import datetime
def join_dates(df: pd.DataFrame):
    """
    Функция объединения близких дат в одну.

    :param df: Таблица данных по одному клиенту
    :return: Таблица данных со схлопнутыми значениями
    """
    df = df.sort_values(by=['date'], ascending=[True])
    unique_dates = pd.DataFrame(df['date'].unique(), columns=['date'])

    # Возвращаем оригинальную таблицу, если была всего одна дата
    if len(unique_dates) == 1:
        return df
    
    unique_dates['new_date'] = unique_dates['date']
    
    dates_list = list(unique_dates['date'])
    for i in range(0, len(dates_list)-1, 2):
        if dates_list[i] == dates_list[i + 1] - timedelta(days=1):
            unique_dates.iat[i, 1] = dates_list[i + 1]
        elif dates_list[i] == dates_list[i + 1] - timedelta(days=2):
            unique_dates.iat[i, 1] = dates_list[i + 1]
    
    new_dates_list = list(unique_dates['new_date'])
    for i in range(len(new_dates_list)-1, 0, -1):
        if new_dates_list[i] == new_dates_list[i - 1] + timedelta(days=1):
            unique_dates.iat[i, 1] = new_dates_list[i - 1]

    df = df.merge(unique_dates, on='date')

    print(unique_dates)
    print(len(list(unique_dates['date'].unique())))
    print(len(list(unique_dates['new_date'].unique())))

    df['date'] = df['new_date']
    
    df = df.groupby(['date', 'product', 'client']).sum().reset_index()
    
    return df

test_df = join_dates(dates_test)

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
    
    
    pool = ThreadPool(8)
    
    df_list = []
    
    def get_df_list(client):

        clients_df = new_df[new_df['client'] == client]
        
        clients_df = join_dates(clients_df)
        
        if clients_df.shape[0] == 1:
            return 1
        
        clients_df = clients_df.sort_values(by=['date', 'sum'], ascending=[True, False])
        
        split = split_client_df(clients_df)
    
        
        if split == 'pass':
            return 1
        
        for s in split:
            df_list.append(s)
            if len(df_list) % 100 == 0:
                print(len(df_list))

    pool.map(get_df_list, clients_list)
    
    pool.close()
    pool.join()
    
    print(len(df_list))
    
    
    for s in df_list:
        products_list = list(s['product'])
        products_list_str = list(map(str, products_list))
        sentences_list.append(products_list_str)
    

    return sentences_list

start = datetime.now()
codes_vect = convert_to_sentences(data=sales,
                            client_column='CardNo',
                            data_column='SaleDate',
                            product_column='Article',
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
model = Word2Vec(codes_vect, workers=4, window=8, sg=0, min_count=min_count, size=130, alpha=0.005, iter=220, compute_loss=True)
print(model.get_latest_training_loss() // 1000000)
model.save('w2v_33m_min1500rub_sg0_i220_window8_size130_during_after')
model.save(r'C:\Users\petr_\Dropbox\Future\w2v_33m_min500rub_sg0_i220_window8_size130_during_after')

#%%

model.save('w2v_mymodel_33_min40_sg0_i250_window7_size130_transSplit')

#%%

def get_similar(lm_codes, topn):
    most_similar = model.wv.most_similar(positive=lm_codes, topn=topn)
    return most_similar


def predict_next_words(lm_codes, topn):
    predict_output_word = model.predict_output_word(lm_codes, topn=topn)
    return predict_output_word


similar = ['15334425']
most_similar = get_similar(similar, topn=15)
print('Артикулы, наиболее похожие на {}'.format(similar))
for ms in most_similar:
    print(ms)

predict = ['15334425']
print('После {} будет: '.format(predict))
predict_output_word = predict_next_words(predict, topn=15)
for pow in predict_output_word:
    print(pow)

loss = model.get_latest_training_loss()
