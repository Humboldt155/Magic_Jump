#%% Импортируем библиотеки и файлы


from datetime import datetime
from gensim.models import Word2Vec
import pandas as pd

model = Word2Vec.load('models/word2vec/w2v_mymodel_33_min50_sg0_i220_window5_size300')

bdd_rms = pd.read_excel('library/BDD.xlsx', names=['product',
                                                   'department',
                                                   'model_adeo',
                                                   'model_name',
                                                   'date_created',
                                                   'product_name',
                                                   'is_stm'])

bdd_rms['product'] = bdd_rms['product'].astype(str)


def get_similar(product: str, num=5, same_model=False):
    """Получить список похожих товаров.
    Parameters:
        product(str): Код продукта
        num(int): Максимальное количество похожих товаров
        same_model(bool): Похожие товары должны быть из одной модели
    Возвращает:
        similars: pd.DataFrame: Таблица похожих товаров
    """

    similars = model.wv.most_similar([product], topn=num*5)

    # Преобразуем
    similars_df = pd.DataFrame(similars, columns=['product', 'probability'])

    # Подтягиваем модель и название
    similars_df = similars_df.merge(bdd_rms, on='product')

    # удаляем текущий артикул, если он попадает в список
    similars_df = similars_df[similars_df['product'] != product]

    # удаляем товары с другими моделями
    if same_model:
        current_model = list(bdd_rms[bdd_rms['product'] == str(product)]['model_adeo'])[0]
        similars_df = similars_df[similars_df['model_adeo'] == current_model]

    similars_df = similars_df.head(num)

    return similars_df


words = pd.DataFrame(list(model.wv.vocab), columns=['product'])

words = words.merge(bdd_rms, on='product')

models_adeo = list(words['model_adeo'].unique())

#%%

import numpy as np

model_adeo = 'MOD_202841'

model_adeo_df = words[words['model_adeo'] == model_adeo]

model_adeo_qty = model_adeo_df.shape[0]

model_adeo_products = list(model_adeo_df['product'].unique())

dfs = []

for product in model_adeo_products:

    similars_df = get_similar(str(product), num=model_adeo_qty, same_model=False)

    similars_df = similars_df.iloc[:, 1:5]
    similars_df_pivot = pd.pivot_table(similars_df, values='probability', index=['model_adeo'], aggfunc=np.sum)
    similars_df_pivot = similars_df_pivot.sort_values('probability', ascending=False)

    i = 0

    not_same = 0
    probable_model = ''

    for index, row in similars_df_pivot.head(2).iterrows():
        main_model = index
        if main_model != model_adeo:
            not_same += 1
            if i == 0:
                model_name = list(similars_df[similars_df['model_adeo'] == main_model]['model_name'])[0]
                probable_model = 'Возможно, товар {} должен быть в модели {} - {}'.format(product, main_model, model_name)
                i += 1

    if not_same == 2:
        print(probable_model)

    dfs.append(similars_df)




