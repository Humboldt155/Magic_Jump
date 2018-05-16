from flask import Flask, render_template

from functions.functions import get_similar

app = Flask(__name__)


def convert_similar(df):
    return df

@app.route('/similar')
def main_similar():
    return render_template(
        'similar.html', **locals())


@app.route('/similar/<string:product>/')
def get_similar(product):
    similar = get_similar(product)
    similar_products = [
        {'model': 'MOD_123', 'products': [{'code': '12345678', 'probability': '0.56'}, {'code': '64788485', 'probability': '0.11'}]},
        {'model': 'MOD_456', 'products': [{'code': '67589879', 'probability': '0.46'}, {'code': '74656373', 'probability': '0.12'}]},
        {'model': 'MOD_987', 'products': [{'code': '22029486', 'probability': '0.46'}, {'code': '74656373', 'probability': '0.12'}]},
        {'model': 'MOD_456', 'products': [{'code': '67589879', 'probability': '0.46'}, {'code': '74656373', 'probability': '0.12'}]},
        {'model': 'MOD_789', 'products': [{'code': '35529952', 'probability': '0.23'}, {'code': '63647485', 'probability': '0.32'}]},
        {'model': 'MOD_546', 'products': [{'code': '35529952', 'probability': '0.23'}, {'code': '63647485', 'probability': '0.32'}]},
        {'model': 'MOD_453', 'products': [{'code': '74656373', 'probability': '0.98'}, {'code': '64623282', 'probability': '0.01'}]},
        {'model': 'MOD_234', 'products': [{'code': '22029486', 'probability': '0.98'}, {'code': '64623282', 'probability': '0.01'}]}
    ]
    return render_template(
        'similar.html', **locals())

if __name__ == '__main__':
    app.run()
