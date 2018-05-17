# Импортируем библиотеки и файлы

from flask import Flask, render_template
from gensim.models import Word2Vec
import pandas as pd

code_model_name = pd.read_excel('library/code_model_name.xlsx',
                                names=['product',
                                       'model_adeo',
                                       'name'])

code_model_name['product'] = code_model_name['product'].astype(str)

model_num = 1  # Какую из моделей использовать

similar_qty = 10  # Количество похожих товаров

models_list = [
    'models/word2vec/w2v_mymodel_33',
    'models/word2vec/w2v_mymodel_33_min50_sg0_i220_window5_size300',
    'models/word2vec/w2v_mymodel_33_min50_sg1_i220_window5_size300',
    'models/word2vec/w2v_mymodel_33_min1000_sg0_i200_window5_size300'
]

model = Word2Vec.load(models_list[model_num])
model_forecast = Word2Vec.load(models_list[3])


#%% Список похожих товаров


def get_similar(product: str, num = 5, same_model = True):
    """Получить список похожих товаров.
    Parameters:
        product(str): Код продукта
        num(int): Максимальное количество похожих товаров
        same_model(bool): Похожие товары должны быть из одной модели
    Возвращает:
        similars: pd.DataFrame: Таблица похожих товаров
    """
    # Список похожих товаров из модели.
    # Умножаем на 20, чтобы увеличить вероятность товаров одной модели
    similars = model.most_similar([product], topn=num*100)

    # Преобразуем
    similars_df = pd.DataFrame(similars, columns=['product', 'probability'])

    # Подтягиваем модель и название
    similars_df = similars_df.merge(code_model_name, on='product')

    # удаляем текущий артикул, если он попадает в список
    similars_df = similars_df[similars_df['product'] != product]

    # удаляем товары с другими моделями
    if same_model:
        current_model = list(code_model_name[code_model_name['product'] == str(product)]['model_adeo'])[0]
        similars_df = similars_df[similars_df['model_adeo'] == current_model]

    similars_df = similars_df.head(num)

    return similars_df

#%% Предсказать покупки


def convert_df(df: pd.DataFrame):
    result = []
    models = list(df['model_adeo'].unique())
    for model in models:
        result.append({'model': model})
    return result


similar = convert_df(get_similar(str(12317232), num=10, same_model=True))

print(similar)


#%%

def get_predicted(products: list, num_codes = 3, num_models = 3, remove_used_models = True):
    """Получить список похожих товаров.
    Parameters:
        products(str): Код продукта
        num_codes(int): Топ моделей
        num_models(int): Топ артикулов в каждой модели
        remove_used_models(bool): Не показывать модели, которые были в запросе
    Возвращает:
        predicted (list): список таблиц pd.DataFrame
    """

    analogs = pd.DataFrame(columns=['product', 'probability', 'model_adeo', 'name'])

    predicted = model.predict_output_word(products, topn=num_codes*num_models*5)
    predicted = pd.DataFrame(predicted, columns=['product', 'probability'])

    similars = []
    for product in products:
        similars.append(list(analogs.append(get_similar(product, num=2, same_model=True))['product']))

    for s in similars:
        for p in s:
            pred = model.predict_output_word([str(p)], topn=num_codes*num_models*5)
            pred = pd.DataFrame(pred, columns=['product', 'probability'])
            pred = pred[pred['product'] != p]
            predicted = predicted.append(pred)


    predicted = predicted.sort_values(by='probability', ascending=False)

    # Удаляем позиции, которые в запросе
    for product in products:
        predicted = predicted[predicted['product'] != product]

    # Подтягиваем модель и название
    predicted = predicted.merge(code_model_name, on='product')

    if remove_used_models:
        current_models = []
        for product in products:
            current_models.append(list(code_model_name[code_model_name['product'] == str(product)]['model_adeo'])[0])
        print(current_models)
        for model_adeo in current_models:
            predicted = predicted[predicted['model_adeo'] != model_adeo]

    predicted = predicted.sort_values(by='probability', ascending=False)

    all_models = list(pd.DataFrame(predicted['model_adeo'].unique())[0])

    result = {}

    for mod in range(0, num_models):
        result[all_models[mod]] = predicted[predicted['model_adeo'] == all_models[mod]].head(num_codes)

    return result

#%%

def get_forecast(products: list, num_codes = 10, num_models = 20, remove_used_models = True):
    """Получить список похожих товаров.
    Parameters:
        products(str): Код продукта
        num_codes(int): Топ моделей
        num_models(int): Топ артикулов в каждой модели
        remove_used_models(bool): Не показывать модели, которые были в запросе
    Возвращает:
        predicted (list): список таблиц pd.DataFrame
    """

    analogs = pd.DataFrame(columns=['product', 'probability', 'model_adeo', 'name'])

    predicted = model_forecast.predict_output_word(products, topn=num_codes*num_models*5)
    predicted = pd.DataFrame(predicted, columns=['product', 'probability'])

    similars = []
    for product in products:
        similars.append(list(analogs.append(get_similar(product, num=5, same_model=True))['product']))

    for s in similars:
        for p in s:
            pred = model_forecast.predict_output_word([str(p)], topn=num_codes*num_models*5)
            pred = pd.DataFrame(pred, columns=['product', 'probability'])
            pred = pred[pred['product'] != p]
            predicted = predicted.append(pred)

    predicted = predicted[predicted['product'].str.find("+") != -1]

    predicted['product'] = predicted['product'].str.slice(0, 8)

    print(predicted)

    predicted = predicted.sort_values(by='probability', ascending=False)

    # Удаляем позиции, которые в запросе
    for product in products:
        predicted = predicted[predicted['product'] != product]

    # Подтягиваем модель и название
    predicted = predicted.merge(code_model_name, on='product')

    if remove_used_models:
        current_models = []
        for product in products:
            current_models.append(list(code_model_name[code_model_name['product'] == str(product)]['model_adeo'])[0])
        print(current_models)
        for model_adeo in current_models:
            predicted = predicted[predicted['model_adeo'] != model_adeo]

    predicted = predicted.sort_values(by='probability', ascending=False)

    return predicted

#%%

print(get_forecast(['13850258']))