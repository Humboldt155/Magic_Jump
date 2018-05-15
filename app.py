from flask import Flask, flash, redirect, render_template, request, session, abort

app = Flask(__name__)



@app.route('/similar')
def main_similar():
    return render_template(
        'similar.html', product='')


@app.route('/similar/<string:product>/')
def get_similar(product):
    return render_template(
        'similar.html', product=product)

if __name__ == '__main__':
    app.run()
