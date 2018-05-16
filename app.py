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


app = Flask(__name__)


def convert_df(df: pd.DataFrame):
    result = []
    models = list(df['model_adeo'].unique())
    for model in models:
        m_dict = {'model': model}
        prods = df[df['model_adeo'] == model]
        products = []
        for index, row in prods.iterrows():
            code = row[0]
            probty = round(row[1] * 100, 10)
            name = row[3]
            products.append({'code': code, 'name': name, 'probability': probty})
        result.append({'model': model, 'products': products[0:8]})

    return result


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

# @app.route('/similar')
# def main_similar():
#     similar_products = convert_df(get_similar(str(12317232)))
#     similar_products = []
#     return render_template(
#         'similar.html', **locals())


@app.route('/similar/<string:product>/')
def get_similar(product):
    name = list(code_model_name[code_model_name['product'] == str(product)]['name'])[0]
    similar_products = convert_df(get_similar(str(product), num=similar_qty, same_model=True))
    relative_products = convert_df_rel(get_similar(str(product), num=200, same_model=False), product)
    return render_template(
        'similar.html', **locals())


@app.route('/predict/<string:product>/')
def get_predicted(product):
    codes_list = product.split(',')
    names = []
    for p in codes_list:
        code = list(code_model_name[code_model_name['product'] == str(p)]['product'])[0]
        name = list(code_model_name[code_model_name['product'] == str(p)]['name'])[0]
        names.append(str(code) + ' - ' + str(name))
    similar_products = convert_df(get_predicted(codes_list, num_codes=similar_qty, num_models=True))
    return render_template(
        'predict.html', **locals())


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

def get_predicted(products: list, num_codes = 5, num_models = 20, remove_used_models = True):
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
    #
    # all_models = list(pd.DataFrame(predicted['model_adeo'].unique())[0])
    #
    # result = {}
    #
    # for mod in range(0, num_models):
    #     result[all_models[mod]] = predicted[predicted['model_adeo'] == all_models[mod]].head(num_codes)

    return predicted
