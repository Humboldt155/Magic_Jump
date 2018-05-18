# Импортируем библиотеки и файлы

from flask import Flask, render_template, jsonify
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

bdd_rms = pd.read_excel('library/BDD.xlsx', names=['product',
                                                   'department',
                                                   'model_adeo',
                                                   'model_name',
                                                   'date_created',
                                                   'product_name',
                                                   'is_stm'])

bdd_rms['product'] = bdd_rms['product'].astype(str)


model = Word2Vec.load(models_list[model_num])
model_forecast = Word2Vec.load(models_list[3])

#%%

app = Flask(__name__)
#%%

#%% API


#  Принимает данные в виде таблицы DF: product, probability, mode_adeo, model_name, date_created, product_name, is_stm
def convert_df_to_json(df: pd.DataFrame, sort_by_stm=False, sort_by_date=False, num_models=10, min_products=10, max_products=10):
    result = {'models': []}
    models = list(df['model_adeo'].unique())  # список всех моделей адево в таблице

    for model_adeo in models:
        model_products = df[df['model_adeo'] == model_adeo]  # фрагмент DataFrame по одной модели
        products = []
        model_name = ''
        product_repeat_check = []  # хранит все коды, для проверки на наличие дублей
        for index, row in model_products.iterrows():
            product = row[0]                       # Код продукта

            # Проверка на наличие повторяющихся позиций
            if product in product_repeat_check:
                continue
            else:
                product_repeat_check.append(product)

            product_name = row[5]                  # Название продукта
            probability = round(row[1] * 100, 10)  # Вероятность
            is_stm = row[6]                        # Собственная торговая марка
            model_name = row[2]                    # Название модели
            date_created = row[4]                  # Дата создания

            # Добавляем атрибуты товара в список товаров
            products.append({'product': product,
                             'product_name': product_name,
                             'probability': probability,
                             'is_stm': is_stm,
                             'date_created': date_created
                             })
        if len(products) < min:
            continue
        result['models'].append({'model_adeo': model_adeo, 'model_name': model_name, 'products': products})

    return result


@app.route('/analogs/<string:product>/')
def get_analogs(product):

    analogs = {'products': convert_df(get_similar(str(product), num=similar_qty, same_model=True))[0]['products']}

    return jsonify(analogs)


#%%  templates

def convert_df(df: pd.DataFrame, min=6):
    result = []
    models = list(df['model_adeo'].unique())
    for model in models:
        prods = df[df['model_adeo'] == model]
        products = []
        code_check = []
        for index, row in prods.iterrows():
            code = row[0]
            if code in code_check:
                continue
            probty = round(row[1] * 100, 10)
            name = row[3]
            code_check.append(code)
            products.append({'code': code, 'name': name, 'probability': probty})
        if len(products) < min:
            continue
        result.append({'model': model, 'products': products[0:6]})

    return result

#%%

def convert_df_rel(df: pd.DataFrame, pr_remove):
    current_model = list(code_model_name[code_model_name['product'] == str(pr_remove)]['model_adeo'])[0]
    result = []
    models = list(df['model_adeo'].unique())
    for model in models:
        if model == current_model:
            continue
        m_dict = {'model': model}
        prods = df[df['model_adeo'] == model]
        products = []
        for index, row in prods.iterrows():
            code = row[0]
            probty = row[1] * 100
            name = row[3]
            products.append({'code': code, 'name': name, 'probability': probty})
        result.append({'model': model, 'products': products[0:5]})

    return result


@app.route('/similars/<string:product>/')
def get_simil(product):
    name = list(code_model_name[code_model_name['product'] == str(product)]['name'])[0]
    similar_products = convert_df(get_similar(str(product), num=similar_qty, same_model=True))
    relative_products = convert_df_rel(get_similar(str(product), num=200, same_model=False), product)
    return render_template(
        'similar.html', **locals())


@app.route('/predict/<string:product>/')
def get_pred(product):
    codes_list = product.split(',')
    names = []
    for p in codes_list:
        code = list(code_model_name[code_model_name['product'] == str(p)]['product'])[0]
        name = list(code_model_name[code_model_name['product'] == str(p)]['name'])[0]
        names.append([str(code), str(name)])
    similar_products = convert_df(get_predicted(codes_list, num_codes=10, num_models=10))
    return render_template(
        'predict.html', **locals())


# @app.route('/tools/<string:product>/')
# def get_too(product):
#     codes_list = product.split(',')
#     names = []
#     for p in codes_list:
#         code = list(code_model_name[code_model_name['product'] == str(p)]['product'])[0]
#         name = list(code_model_name[code_model_name['product'] == str(p)]['name'])[0]
#         names.append([str(code), str(name)])
#     similar_products = convert_df(get_tools(codes_list, num=600), min=1)
#     return render_template(
#         'tools.html', **locals())
#
#
# if __name__ == '__main__':
#     app.run()


@app.route('/forecast/<string:product>/')
def get_forec(product):
    codes_list = product.split(',')
    names = []
    for p in codes_list:
        code = list(code_model_name[code_model_name['product'] == str(p)]['product'])[0]
        name = list(code_model_name[code_model_name['product'] == str(p)]['name'])[0]
        names.append([str(code), str(name)])
    similar_products = convert_df(get_forecast(codes_list, num_codes=10, num_models=10))
    return render_template(
        'forecast.html', **locals())


if __name__ == '__main__':
    app.run()


# функции

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

def get_predicted(products: list, num_codes = 6, num_models = 20, remove_used_models = True):
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
        similars.append(list(analogs.append(get_similar(product, num=4, same_model=True))['product']))

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

    # Удалить дубликаты

    #
    # all_models = list(pd.DataFrame(predicted['model_adeo'].unique())[0])
    #
    # result = {}
    #
    # for mod in range(0, num_models):
    #     result[all_models[mod]] = predicted[predicted['model_adeo'] == all_models[mod]].head(num_codes)

    return predicted


#%%

def get_forecast(products: list, num_codes = 10, num_models = 10, remove_used_models = True):
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

#%% Список инструментов и аксессуаров

#
# def get_tools(products: list, num=1000):
#     """Получить список инструментов.
#     Parameters:
#         products(list): Список товаров
#         num(int): Топ моделей
#     Возвращает:
#         predicted (list): список таблиц pd.DataFrame
#     """
#
#     analogs = pd.DataFrame(columns=['product', 'probability', 'model_adeo', 'name'])
#
#     predicted = model_forecast.predict_output_word(products, topn=num)
#     predicted = pd.DataFrame(predicted, columns=['product', 'probability'])
#
#     similars = []
#     for product in products:
#         similars.append(list(analogs.append(get_similar(product, num=4, same_model=True))['product']))
#
#     for s in similars:
#         for p in s:
#             pred = model_forecast.predict_output_word([str(p)], topn=4)
#             pred = pd.DataFrame(pred, columns=['product', 'probability'])
#             pred = pred[pred['product'] != p]
#             predicted = predicted.append(pred)
#
#     predicted = predicted[predicted['product'].str.find("+") != -1]
#
#     predicted['product'] = predicted['product'].str.slice(0, 8)
#
#     predicted = predicted.sort_values(by='probability', ascending=False)
#
#     # Удаляем позиции, которые в запросе
#     for product in products:
#         predicted = predicted[predicted['product'] != product]
#
#     # Подтягиваем модель и название
#     predicted = predicted.merge(code_model_name, on='product')
#
#     predicted = predicted.merge(tool_models, on='model_adeo')
#
#     predicted = predicted.iloc[:, 0:4]
#
#     #predicted = predicted[predicted['model_adeo'] == 'MOD_20']
#
#     predicted = predicted.sort_values(by='probability', ascending=False)
#
#     return predicted

#%%
#similar_products = convert_df(get_tools(['12317232'], num=600), min=1)
