#%% Импортируем библиотеки и модели

from gensim.models import Word2Vec
import pandas as pd

code_model_name = pd.read_excel('library/code_model_name.xlsx',
                                names=['product',
                                       'model_adeo',
                                       'name'])

code_model_name['product'] = code_model_name['product'].astype(str)

model = Word2Vec.load('models/word2vec/w2v_mymodel_33')


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
    print('Список похожих артикулов загружен: {} штук'.format(len(similars)))

    # Преобразуем
    similars_df = pd.DataFrame(similars, columns=['product', 'probability'])
    print('Список конвертирован в DataFrame: {} штук'.format(len(similars_df)))

    # Подтягиваем модель и название
    similars_df = similars_df.merge(code_model_name, on='product')
    print('Модель и название загружены: {} штук'.format(len(similars_df)))

    # удаляем текущий артикул, если он попадает в список
    similars_df = similars_df[similars_df['product'] != product]
    print('Текущий артикул удален: {} штук'.format(len(similars_df)))

    # удаляем товары с другими моделями
    if same_model:
        current_model = list(code_model_name[code_model_name['product'] == str(product)]['model_adeo'])[0]
        print('Текущая модель: {}'.format(current_model))
        similars_df = similars_df[similars_df['model_adeo'] == current_model]
        print('Список очищен от других моделей: {} штук'.format(len(similars_df)))

    similars_df = similars_df.head(num)
    print('Конечный список готов: {} штук'.format(len(similars_df)))

    return similars_df

#%% Тест


print(get_similar('18669554', num=10, same_model=True))

#%% Предсказать покупки


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


predicted = get_predicted(['15317297'], num_codes=2, num_models = 20, remove_used_models = True)

for key in predicted:
    print('')
    print('')
    print('')
    print('Модель: {}'.format(key))
    print(predicted[key])
