#%%

import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics


model = Word2Vec.load('models/word2vec/w2v_mymodel_33_min5_sg0_i250_window3_size300_transSplit')

bdd_rms = pd.read_excel('library/BDD.xlsx', names=['product',
                                                   'department',
                                                   'model_adeo',
                                                   'model_name',
                                                   'date_created',
                                                   'product_name',
                                                   'is_stm'])

word_vectors = KeyedVectors.load('models/word2vec/w2v_mymodel_33_min5_sg0_i250_window3_size300_transSplit')

#%%


def get_model_vocab(model_adeo: str):

    all_products_df = pd.DataFrame(list(model.wv.vocab), columns=['product'])

    all_products_df['product'] = all_products_df['product'].astype(int)
    all_products_df = all_products_df.merge(bdd_rms, on='product')

    model_products_df = all_products_df[all_products_df['model_adeo'] == model_adeo]

    products_list = list(model_products_df['product'])

    vecs = np.array(list(word_vectors.wv.get_vector(str(prod)) for prod in products_list))

    return [vecs, model_products_df]


def list_to_X(products: list):

    vectors = np.array(list(word_vectors.wv.get_vector(str(prod)) for prod in products))

    return vectors


predict_boiler = list(pd.DataFrame(model.predict_output_word(['12317240'], topn=100))[0])

X = list_to_X(predict_boiler)


#%%

model_adeo = 'MOD_200873'
max_clusters = 10
num_clusters = 7

wcss = []
for i in range(1, max_clusters):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, max_clusters), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = num_clusters, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
labels = list(kmeans.labels_)
centroids = kmeans.cluster_centers_

print("Cluster id labels for inputted data")
print(labels)
# print("Centroids data")
# print(centroids)

products_df = pd.DataFrame(predict_boiler, columns=['product']).astype(int)
products_df = products_df.merge(bdd_rms, on='product')

products_df['cluster'] = labels

clustered_products = products_df.sort_values(by=['cluster'])

