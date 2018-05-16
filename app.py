# Импортируем библиотеки и файлы

from flask import Flask, render_template
from gensim.models import Word2Vec
import pandas as pd

code_model_name = pd.read_excel('library/code_model_name.xlsx',
                                names=['product',
                                       'model_adeo',
                                       'name'])

code_model_name['product'] = code_model_name['product'].astype(str)
model = Word2Vec.load('models/word2vec/w2v_mymodel_33')


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
            probty = row[1]
            name = row[3]
            products.append({'code': code, 'name': name, 'probability': probty})
        result.append({'model': model, 'products': products[0:6]})

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
            probty = row[1]
            name = row[3]
            products.append({'code': code, 'name': name, 'probability': probty})
        result.append({'model': model, 'products': products[0:6]})

    return result

@app.route('/similar')
def main_similar():
    similar_products = convert_df(get_similar(str(12317232)))
    return render_template(
        'similar.html', **locals())


@app.route('/similar/<string:product>/')
def get_similar(product):
    similar_products = convert_df(get_similar(str(product), num=8, same_model=True))
    relative_products = convert_df_rel(get_similar(str(product), num=200, same_model=False), product)
    return render_template(
        'similar.html', **locals())


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
