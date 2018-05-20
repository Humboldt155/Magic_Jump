# Импортируем библиотеки и файлы
from datetime import datetime

from flask import Flask, render_template, jsonify
from gensim.models import Word2Vec
import pandas as pd
from flask_cors import CORS

model_num = 0 # Какую из моделей использовать

similar_qty = 10  # Количество похожих товаров

models_list = [
    'models/word2vec/w2v_mymodel_33_min50_sg0_i220_window5_size300',               #
    'models/word2vec/w2v_mymodel_33_min1000_sg0_i200_window5_size300',             # ------------
    'models/word2vec/w2v_mymodel_33_min5_sg0_i250_window3_size300_transSplit',     #
    'models/word2vec/w2v_mymodel_33_min500_sg0_i220_window10_size300_transSplit',  # ------------
    'models/word2vec/w2v_mymodel_33_mincount1_min1_sg0_i230_window5_size300',      #
    'models/word2vec/w2v_mymodel_33_mincount1_min1_sg0_i400_window10_size300',     #
]

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


model = Word2Vec.load(models_list[model_num])
model_forecast = Word2Vec.load(models_list[5])

app = Flask(__name__)
CORS(app)


#%% API

@app.route('/analogs/<string:product>/')
def get_analogs(product):

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

    return jsonify(analogs)


@app.route('/complementary/<string:products>/')
def get_complementary(products):

    products_list = products.split(',')
    main_product = products_list[0]

    product_name = list(bdd_rms[bdd_rms['product'] == str(main_product)]['product_name'])[0]

    #  Получаем датафрейм с аналогами
    complementary_df_main = get_predicted(products_list, num_models=5, num_products=5)

    #  Сортируем и отбираем данные, возвращаем датафрейм
    complementary_df_cut_sort = cut_and_sort(complementary_df_main,
                                       min_products=3,
                                       max_products=5,
                                       num_models=5,
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

    start_time = datetime.now()

    analogs = pd.DataFrame(columns=['product', 'probability', 'model_adeo', 'name'])

    predicted = model_forecast.predict_output_word(products, topn=num_codes*num_models*3)
    predicted = pd.DataFrame(predicted, columns=['product', 'probability'])

    similars = []
    for product in products[0]:
        similars.append(list(analogs.append(get_similar(product, num=3, same_model=True))['product']))

    for product in products[0]:
        similars.append(list(analogs.append(get_similar(product, num=3, same_model=True))['product']))

    for s in similars:
        for p in s:
            pred = model_forecast.predict_output_word([str(p)], topn=num_codes*num_models*5)
            pred = pd.DataFrame(pred, columns=['product', 'probability'])
            pred = pred[pred['product'] != p]
            predicted = predicted.append(pred)
            print(p)
            print(s)

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
        for model_adeo in current_models:
            predicted = predicted[predicted['model_adeo'] != model_adeo]

    predicted = predicted.sort_values(by='probability', ascending=False)

    return predicted


#%%


if __name__ == '__main__':
    app.run()


#%% Тест

#print(get_similar('12317232', num=8, same_model=True))

# for i in range(0, 6):
#     model = Word2Vec.load(models_list[i])
#     similars = get_similar('14154254', num=150, same_model=True)
#     print(len(similars))
