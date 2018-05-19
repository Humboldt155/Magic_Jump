# Импортируем библиотеки и файлы

from flask import Flask, render_template, jsonify
from gensim.models import Word2Vec
import pandas as pd
from flask_cors import CORS

model_num = 1  # Какую из моделей использовать

similar_qty = 10  # Количество похожих товаров

models_list = [
    'models/word2vec/w2v_mymodel_33',
    'models/word2vec/w2v_mymodel_33_min50_sg0_i220_window5_size300',
    'models/word2vec/w2v_mymodel_33_min50_sg1_i220_window5_size300',
    'models/word2vec/w2v_mymodel_33_min1000_sg0_i200_window5_size300',
    'models/word2vec/w2v_mymodel_33_min5_sg0_i250_window3_size300_transSplit'
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
CORS(app)
#%%

#%% API


#  Принимает данные в виде таблицы DF: product, probability, mode_adeo, model_name, date_created, product_name, is_stm
def convert_df_to_json(df: pd.DataFrame):
    """Конвертировать DataFrame в формат json.
    Параметры:
        df(pd.DataFrame): Таблица данных, полученная в результате предсказания
        sort_by_stm(bool): на первом месте будут товары СТМ, если они есть
        sort_by_date(bool): товары отсортированы по новизне. Если выбран sort_by_stm, то сначала новинки СТМ
        num_models(int): Максимальное число моделей к выдаче
        min_products(int): Минимальное число товаров, которое модет быть в одной модели
        min_products(int): Максимальное число товаров, которое модет быть в одной модели
        increase_for_stm(bool): Насколько предварительно увеличить выборку, чтобы повысить шанс товаров СТМ
    Возвращает:
        predicted (list): список таблиц pd.DataFrame
    """
    result = {'models': []}

    #  список всех встречающихся моделей
    models = list(df['model_adeo'].unique())  # список всех моделей адево в таблице
    assert len(models) > 0, 'Список моделей пуст'

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

            product_name = row[6]                  # Название продукта
            probability = round(row[1] * 100, 10)  # Вероятность
            is_stm = row[7]                        # Собственная торговая марка
            model_name = row[4]                    # Название модели
            date_created = row[5]                  # Дата создания

            # Добавляем атрибуты товара в список товаров
            products.append({'product': product,
                             'product_name': product_name,
                             'probability': probability,
                             'is_stm': is_stm,
                             'date_created': date_created
                             })

        result['models'].append({'model_adeo': model_adeo,
                                 'model_name': model_name,
                                 'products': products})

    return result

#%%


@app.route('/analogs/<string:product>/')
def get_analogs(product):

    #  Получаем датафрейм с аналогами
    analogs_df_main = get_similar(str(product), num=8, same_model=True)

    #  Сортируем и отбираем данные, возвращаем датафрейм
    analogs_df_cut_sort = cut_and_sort(analogs_df_main,
                                       min_products=1,
                                       max_products=8,
                                       sort_by_stm=True,
                                       sort_by_date=False)

    #  Конвертируем в формат json
    analogs = convert_df_to_json(analogs_df_cut_sort)

    return jsonify(analogs)


@app.route('/complementary/<string:products>/')
def get_complementary(products):

    products_list = products.split(',')

    main_product = products_list[0]

    product_name = list(bdd_rms[bdd_rms['product'] == str(main_product)]['product_name'])[0]

    #  Получаем датафрейм с аналогами
    analogs_df_main = get_predicted(products_list, num_models=5, num_products=5)

    #  Сортируем и отбираем данные, возвращаем датафрейм
    analogs_df_cut_sort = cut_and_sort(analogs_df_main,
                                       min_products=3,
                                       max_products=5,
                                       num_models=5,
                                       sort_by_stm=False,
                                       sort_by_date=False)

    #  Конвертируем в формат json
    analogs = convert_df_to_json(analogs_df_cut_sort)

    analogs.update({'product': main_product})
    analogs.update({'product_name': product_name})

    return jsonify(analogs)


#%%

# @app.route('/predict/<string:product>/')
# def get_pred(product):
#     codes_list = product.split(',')
#     names = []
#     for p in codes_list:
#         code = list(code_model_name[code_model_name['product'] == str(p)]['product'])[0]
#         name = list(code_model_name[code_model_name['product'] == str(p)]['name'])[0]
#         names.append([str(code), str(name)])
#     similar_products = convert_df(get_predicted(codes_list, num_codes=10, num_models=10))
#     return render_template(
#         'predict.html', **locals())


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
if __name__ == '__main__':
    app.run()


# @app.route('/forecast/<string:product>/')
# def get_forec(product):
#     codes_list = product.split(',')
#     names = []
#     for p in codes_list:
#         code = list(code_model_name[code_model_name['product'] == str(p)]['product'])[0]
#         name = list(code_model_name[code_model_name['product'] == str(p)]['name'])[0]
#         names.append([str(code), str(name)])
#     similar_products = convert_df(get_forecast(codes_list, num_codes=10, num_models=10))
#     return render_template(
#         'forecast.html', **locals())
#
#
# if __name__ == '__main__':
#     app.run()


#%%

def cut_and_sort(df: pd.DataFrame,
                 min_products=4,
                 max_products=10,
                 sort_by_stm=False,
                 sort_by_date=False,
                 num_models=10,
                 chance_increase=0):

    assert chance_increase >= 0, 'Увеличение вероятности появления новинки или СТМ не может быть ниже 0'

    models_list = df['model_adeo'].unique()

    chance_qty = int(max_products * (1 + chance_increase))

    models_pack = []

    for model in models_list:
        model_df = df[df['model_adeo'] == model]

        if model_df.shape[0] < min_products:
            continue

        model_df = model_df.iloc[0:chance_qty, :]

        if sort_by_date:
            model_df = model_df.sort_values(by='date_created', ascending=False)

        if sort_by_stm:
            model_df = model_df.sort_values(by='is_stm', ascending=False)

        model_df = model_df.iloc[0:max_products, :]

        models_pack.append(model_df)

    result_df = pd.concat(models_pack[0:num_models])

    return result_df


#%% Список похожих товаров


def get_similar(product: str, num=5, same_model=True):
    """Получить список похожих товаров.
    Parameters:
        product(str): Код продукта
        num(int): Максимальное количество похожих товаров
        same_model(bool): Похожие товары должны быть из одной модели
    Возвращает:
        similars: pd.DataFrame: Таблица похожих товаров
    """

    similars = model.most_similar([product], topn=num*20)

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

    similars_df = similars_df

    return similars_df


#%% Предсказать покупки

def get_predicted(products: list, num_models=10, num_products=10, remove_used_models=True):
    """Получить список похожих товаров.
    Parameters:
        products(list): Код продукта
        num_products(int): Топ моделей
        num_models(int): Топ артикулов в каждой модели
        remove_used_models(bool): Не показывать модели, которые были в запросе
    Возвращает:
        predicted (list): список таблиц pd.DataFrame
    """

    analogs = pd.DataFrame(columns=['product', 'probability', 'model_adeo', 'name'])

    predicted = model.predict_output_word(products, topn=num_products*num_models*2)
    predicted = pd.DataFrame(predicted, columns=['product', 'probability'])

    similars = []
    for product in products:
        similars.append(list(analogs.append(get_similar(product, num=2, same_model=True))['product']))

    for s in similars:
        for p in s:
            pred = model.predict_output_word([str(p)], topn=num_products*num_models)
            pred = pd.DataFrame(pred, columns=['product', 'probability'])
            pred = pred[pred['product'] != p]
            predicted = predicted.append(pred)

    predicted = predicted.sort_values(by='probability', ascending=False)

    # Удаляем позиции, которые в запросе
    for product in products:
        predicted = predicted[predicted['product'] != product]

    # Подтягиваем модель и название
    predicted = predicted.merge(bdd_rms, on='product')

    if remove_used_models:
        current_models = []
        for product in products:
            current_models.append(list(bdd_rms[bdd_rms['product'] == str(product)]['model_adeo'])[0])
        print(current_models)
        for model_adeo in current_models:
            predicted = predicted[predicted['model_adeo'] != model_adeo]

    predicted = predicted.sort_values(by='probability', ascending=False)

    predicted = predicted.drop_duplicates(subset='product', keep='first')

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

    predicted = model_forecast.predict_output_word(products, topn=num_codes*num_models*3)
    predicted = pd.DataFrame(predicted, columns=['product', 'probability'])

    similars = []
    for product in products[0]:
        similars.append(list(analogs.append(get_similar(product, num=3, same_model=True))['product']))

    for s in similars:
        for p in s:
            pred = model_forecast.predict_output_word([str(p)], topn=num_codes*num_models*5)
            pred = pd.DataFrame(pred, columns=['product', 'probability'])
            pred = pred[pred['product'] != p]
            predicted = predicted.append(pred)

    predicted = predicted[predicted['product'].str.find("+") != -1]

    predicted['product'] = predicted['product'].str.slice(0, 8)

    predicted = predicted



    # Удаляем позиции, которые в запросе
    for product in products:
        predicted = predicted[predicted['product'] != product]

    # Подтягиваем модель и название
    predicted = predicted.merge(bdd_rms, on='product')

    if remove_used_models:
        current_models = []
        for product in products:
            current_models.append(list(bdd_rms[bdd_rms['product'] == str(product)]['model_adeo'])[0])
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


# similar_products = get_similar(str('81946088'), num=50, same_model=True)
# similar_products_conv = convert_df_to_json(similar_products)


