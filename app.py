# Импортируем библиотеки и файлы
from datetime import datetime
from json import dumps

from flask import Flask, render_template, jsonify
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

from flask_cors import CORS

model_num = 0 # Какую из моделей использовать

similar_qty = 10  # Количество похожих товаров

models_list = [
    'models/word2vec/w2v_33m_min20rub_sg0_i220_window7_size130',
    'models/word2vec/w2v_33m_min500rub_sg0_i220_window8_size130_during_after',
]

model_categorical = Word2Vec.load('models/word2vec/w2v_33m_categorical')

# Общие требования: sg1 - не использовать
# Скорость обучения и количество эпох (количество) - и скорость 0,005 пока лучший результат 220 достаточно


bdd_rms = pd.read_excel('library/BDD.xlsx', names=['product',
                                                   'department',
                                                   'model_adeo',
                                                   'model_name',
                                                   'date_created',
                                                   'product_name',
                                                   'is_stm'])

bdd_rms['product'] = bdd_rms['product'].astype(str)


model = Word2Vec.load(models_list[0])
model_forecast = Word2Vec.load(models_list[1])
model_clusters = Word2Vec.load(models_list[0])
word_vectors = KeyedVectors.load(models_list[0])


app = Flask(__name__)
CORS(app)



#%% API


def predict_categories(product, num):
    model_adeo = bdd_rms[bdd_rms['product'] == product]['model_adeo'].values[0]
    prediction = pd.DataFrame(model_categorical.predict_output_word([model_adeo], topn=num),
                              columns=['model_adeo', 'probability'])

    prediction = prediction.merge(bdd_rms, on='model_adeo', how='inner')
    prediction = prediction.drop_duplicates(subset='model_adeo', keep='first')

    prediction = prediction.to_dict(('records'))

    return prediction

#print(predict_categories('18743611'))


#%%

@app.route('/categorical_predict/<string:product>/<int:qty>')
def get_predict_categories(product, qty):
    predict = predict_categories(product, qty)

    return jsonify(predict)


#%%

@app.route('/most_similar/<string:product>/<int:qty>')
def get_most_similar(product, qty):

    start_time = datetime.now()

    #  Получаем датафрейм с аналогами
    most_similar = model.most_similar([str(product)], topn=qty)
    model_time = datetime.now() - start_time
    print(model_time)
    model_time = str('{},{} секунд'.format(model_time.seconds, "%06d" % model_time.microseconds))

    similars_df = pd.DataFrame(most_similar, columns=['product', 'probability'])

    # Подтягиваем модель и название
    similars_df = similars_df.merge(bdd_rms, on='product')

    #  Конвертируем в формат json
    similars_dict = similars_df.to_dict('records')

    all_time = datetime.now() - start_time
    all_time = str('{},{} секунд'.format(all_time.seconds, "%06d" % all_time.microseconds))

    return jsonify({'similars': similars_dict, 'model_time': model_time, 'all_time': all_time})


#%% Analogs

@app.route('/analogs/<string:product>/')
def get_analogs(product):

    start_time = datetime.now()

    #  Получаем датафрейм с аналогами
    analogs_df_main = get_similar(str(product), num=5, same_model=True)

    #  Сортируем и отбираем данные, возвращаем датафрейм
    analogs_df_cut_sort = cut_and_sort(analogs_df_main,
                                       min_products=1,
                                       max_products=8,
                                       sort_by_stm=True,
                                       sort_by_date=False)

    #  Конвертируем в формат json
    analogs = convert_df_to_json(analogs_df_cut_sort)

    print('Аналоги, найдены за {}'.format(datetime.now() - start_time))

    return jsonify(analogs)


@app.route('/complementary/<string:products>/')
def get_complementary(products):

    products_list = products.split(',')
    main_product = products_list[0]

    product_name = list(bdd_rms[bdd_rms['product'] == str(main_product)]['product_name'])[0]

    #  Получаем датафрейм с аналогами
    complementary_df_main = get_predicted(products_list, num_models=7, num_products=5)

    #  Сортируем и отбираем данные, возвращаем датафрейм
    complementary_df_cut_sort = cut_and_sort(complementary_df_main,
                                       min_products=1,
                                       max_products=3,
                                       num_models=7,
                                       sort_by_stm=False,
                                       sort_by_date=False)

    #  Конвертируем в формат json
    complementary = convert_df_to_json(complementary_df_cut_sort)

    complementary.update({'product': main_product})
    complementary.update({'product_name': product_name})

    return jsonify(complementary)


@app.route('/supplementary/<string:products>/')
def get_supplementary(products):

    products_list = products.split(',')

    #  Получаем датафрейм с аналогами
    supplementary_df_main = get_supplement(products_list, num_models=5, num_products=5)

    #  Сортируем и отбираем данные, возвращаем датафрейм
    supplementary_df_cut_sort = cut_and_sort(supplementary_df_main,
                                       min_products=3,
                                       max_products=5,
                                       num_models=5,
                                       sort_by_stm=False,
                                       sort_by_date=False)

    #  Конвертируем в формат json
    supplementary = convert_df_to_json(supplementary_df_cut_sort)

    return jsonify(supplementary)

#%% Получить прогноз покупок
@app.route('/forecast/<string:products>/')
def get_forec(products):

    products_list = products.split(',')

    #  Получаем датафрейм с аналогами
    forecast_during_df, forecast_after_df = get_forecast(products_list)

    forecast_during = forecast_during_df.to_dict('records')
    forecast_after = forecast_after_df.to_dict('records')

    return_obj = {'forecast_during': forecast_during, 'forecast_after': forecast_after}


    return jsonify(return_obj)


#%%  Конвертировать в json


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


#%%  Получить векторное представление по номеру артикула

@app.route('/cluster/<string:product>/<int:clusters>/<int:num_products>/<int:products_for_cluster>')
def get_clust(product, clusters, num_products, products_for_cluster):

    if products_for_cluster * clusters > num_products:
        num_products = products_for_cluster * clusters

    predict = list(pd.DataFrame(model.predict_output_word([product], topn=num_products))[0])

    predict_df = pd.DataFrame(model.predict_output_word([product], topn=num_products), columns=['product', 'probability'])
    predict_list = list(predict_df['product'])

    X = list_to_X(predict_list)

    kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=42)
    kmeans.fit_predict(X)
    labels = list(kmeans.labels_)
    products_df = predict_df.merge(bdd_rms, on='product')


    products_df['cluster'] = labels
    products_df = products_df.sort_values(by='probability', ascending=False)


    cluster_list = list(products_df['cluster'].unique())

    clusters_list = []
    for cl in cluster_list:
        cluster_df = products_df[products_df['cluster'] == cl]

        cluster_dict = cluster_df.to_dict('records')
        clusters_list.append({'cluster_num': str(cl + 1), 'products': cluster_dict})

    return jsonify({'clusters': clusters_list})

#X = get_clust('18745342')


#%%  Получить векторное представление по номеру артикула


def list_to_X(products: list):
    vectors = np.array(list(word_vectors.wv.get_vector(str(prod)) for prod in products))
    return vectors


predict = list(pd.DataFrame(model.predict_output_word(['12317240'], topn=100))[0])


#%% Обрезать и отсортировать

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

    similars = model.most_similar([product], topn=num*5)

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
    start_time = datetime.now()
    analogs = pd.DataFrame(columns=['product', 'probability', 'model_adeo', 'name'])

    predicted = model.predict_output_word(products, topn=(len(products) + num_products)*num_models*2)

    print('Получено предсказание: {}'.format(datetime.now() - start_time))
    new_time = datetime.now()

    predicted = pd.DataFrame(predicted, columns=['product', 'probability'])

    print('Предсказание конвертировано в DF: {}'.format(datetime.now() - new_time))
    new_time = datetime.now()

    similars = []
    for product in products[0:1]:
        similars.append(list(analogs.append(get_similar(product, num=3, same_model=True))['product']))

    print('Получены похожие товары в DF: {}'.format(datetime.now() - new_time))
    new_time = datetime.now()

    for s in similars:
        for p in s:
            pred = model.predict_output_word([str(p)], topn=(len(products) + num_products)*num_models)
            pred = pd.DataFrame(pred, columns=['product', 'probability'])
            pred = pred[pred['product'] != p]
            predicted = predicted.append(pred)
            print(p)
        print(s)

    print('Получены предсказания для похожих: {}'.format(datetime.now() - new_time))
    new_time = datetime.now()

    predicted = predicted.sort_values(by='probability', ascending=False)

    print('Данные отсортированы: {}'.format(datetime.now() - new_time))
    new_time = datetime.now()

    # Удаляем позиции, которые в запросе
    for product in products:
        predicted = predicted[predicted['product'] != product]

    print('Удалены лишние позиции: {}'.format(datetime.now() - new_time))
    new_time = datetime.now()

    # Подтягиваем модель и название
    predicted = predicted.merge(bdd_rms, on='product')

    print('Подтянуты столбцы: {}'.format(datetime.now() - new_time))
    new_time = datetime.now()

    if remove_used_models:
        current_models = []
        for product in products:
            current_models.append(list(bdd_rms[bdd_rms['product'] == str(product)]['model_adeo'])[0])
        for model_adeo in current_models:
            predicted = predicted[predicted['model_adeo'] != model_adeo]

    print('Удалены действующие модели {}'.format(datetime.now() - new_time))
    new_time = datetime.now()

    predicted = predicted.sort_values(by='probability', ascending=False)

    print('Отсортированы {}'.format(datetime.now() - new_time))
    new_time = datetime.now()

    predicted = predicted.drop_duplicates(subset='product', keep='first')

    print('Удалены дубликаты {}'.format(datetime.now() - new_time))
    new_time = datetime.now()
    print('Всего затрачено времени {}'.format(datetime.now() - start_time))

    return predicted

#%% Возможно, клиент забыл купить


def get_supplement(products: list, num_models=5, num_products=3, remove_used_models=True):
    """Получить список товаров, которые клиент, возможно, забыл приобрести.
    Parameters:
        products(list): Код продукта
        num_products(int): Топ моделей
        num_models(int): Топ артикулов в каждой модели
        remove_used_models(bool): Не показывать модели, которые были в запросе
    Возвращает:
        predicted (list): список таблиц pd.DataFrame
    """
    analogs = pd.DataFrame(columns=['product', 'probability', 'model_adeo', 'name'])
    predicted = pd.DataFrame(columns=['product', 'probability'])

    # Получаем похожий товар для каждого товара в списке
    similars = []
    for product in products:
        similars.append(list(analogs.append(get_similar(product, num=2, same_model=True))['product']))

    # Добавляем товар в запросе
    similars.append(products)

    # Проходим по всем товарам циклом и определяем предсказание для каждого в отдельности
    for s in similars:
        for p in s:
            pred = model.predict_output_word([str(p)], topn=(len(products) + num_products)*num_models)
            pred = pd.DataFrame(pred, columns=['product', 'probability'])
            pred = pred[pred['product'] != p]
            predicted = predicted.append(pred)
            print(p)
        print(s)

    predicted = predicted.sort_values(by='probability', ascending=False)

    # Удаляем дубликаты артикулов
    predicted = predicted.drop_duplicates(subset='product', keep='first')

    # Удаляем позиции, которые в запросе
    for product in products:
        predicted = predicted[predicted['product'] != product]

    # Подтягиваем модель и название
    predicted = predicted.merge(bdd_rms, on='product')

    # Удаляем использованные модели
    if remove_used_models:
        current_models = []
        for product in products:
            current_models.append(list(bdd_rms[bdd_rms['product'] == str(product)]['model_adeo'])[0])
        for model_adeo in current_models:
            predicted = predicted[predicted['model_adeo'] != model_adeo]

    predicted = predicted.sort_values(by='probability', ascending=False)

    return predicted


#%% Прогноз будущих покупок

def get_forecast(products: list, num_models=10, num_products=10, remove_used_models=True):
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

    predicted = model_forecast.predict_output_word(products, topn=1000)

    predicted = pd.DataFrame(predicted, columns=['product', 'probability'])

    predicted = predicted.sort_values(by='probability', ascending=False)

    # Удаляем позиции, которые в запросе
    for product in products:
        predicted = predicted[predicted['product'] != product]

    # Удаляем позиции без знака +

    predicted = predicted[predicted['product'].str.slice(0, 8) != predicted['product']]

    predicted_during = predicted[predicted['product'].str.slice(8, 10) == '+'].copy()
    predicted_during['product'] = predicted_during['product'].str.slice(0, 8)
    predicted_after = predicted[predicted['product'].str.slice(8, 10) == '++'].copy()
    predicted_after['product'] = predicted_after['product'].str.slice(0, 8)

    # Подтягиваем модель и название
    predicted_during = predicted_during.merge(bdd_rms, on='product')
    predicted_after = predicted_after.merge(bdd_rms, on='product')

    # print(predicted)

    if remove_used_models:
        current_models = []
        for product in products:
            current_models.append(list(bdd_rms[bdd_rms['product'] == str(product)]['model_adeo'])[0])
        for model_adeo in current_models:
            predicted_during = predicted_during[predicted_during['model_adeo'] != model_adeo]
            predicted_after = predicted_after[predicted_after['model_adeo'] != model_adeo]

    predicted_during = predicted_during.sort_values(by='probability', ascending=False)
    predicted_during = predicted_during.drop_duplicates(subset='product', keep='first')

    predicted_after = predicted_after.sort_values(by='probability', ascending=False)
    predicted_after = predicted_after.drop_duplicates(subset='product', keep='first')

    return [predicted_during, predicted_after]


# test_forecast = get_forecast(['81953451'])


#%%


if __name__ == '__main__':
    app.run()


#%% Тест

#print(get_similar('12317232', num=8, same_model=True))

# for i in range(0, 6):
#     model = Word2Vec.load(models_list[i])
#     similars = get_similar('14154254', num=150, same_model=True)
#     print(len(similars))
