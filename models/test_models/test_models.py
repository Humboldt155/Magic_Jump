#%% Импортируем библиотеки и файлы
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
    'models/word2vec/w2v_mymodel_33_min1000_sg0_i200_window5_size300',  # ------------
    'models/word2vec/w2v_mymodel_33_mincount1_min1_sg0_i230_window5_size300',      #
    # 'models/word2vec/w2v_mymodel_33_mincount1_min1_sg0_i400_window10_size300',     #
    'models/word2vec/test_29 of 81 nodes_250 alpha_ 0.0025 epochs_ 100 windows_ 4 time_0_02_20_155266'
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

#%%


