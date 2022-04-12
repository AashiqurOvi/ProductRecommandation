# -*- coding: utf-8 -*-
"""
Created on Tue Nov 9 10:31:57 2021

@author: Ovi
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for
import numpy as np
import pandas as pd
import pickle
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors






# define a function that creates nearest neighbour
# if it doesn't exist
def create_nn():
    data = pickle.load(open('product_features_df.pkl','rb'))
    product_features_df_matrix = csr_matrix(data.values)
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(product_features_df_matrix)
    return data,model_knn


# defining a function that recommends 10 most similar movies
def rcmd(p,n):
    data = pickle.load(open('product_features_df.pkl','rb'))
    product_features_df_matrix = csr_matrix(data.values)
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(product_features_df_matrix)
    # check if data and sim are already assigned
    try:
        data.head()
    except:
        data, model_knn = create_nn()
    # check if the product is in our database or not
    productList= list(data.index)
    if p not in productList:
        return('This product is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the product in the dataframe
        query_index = data.index.get_loc(p)

        # taking top 1- product scores
        # not taking the first index since it is the same product
        distances, indices = model_knn.kneighbors(data.iloc[query_index,:].values.reshape(1, -1), n_neighbors = int(n)+1 )       
        

        # making an empty list that will containg n numbr of product recommendations
        l = []
        for i in range(1, len(distances.flatten())):
            l.append(data.index[indices.flatten()[i]])
        return l

app = Flask(__name__)

@app.route("/")
def home():
    data = pickle.load(open('product_features_df.pkl','rb'))
    return render_template('home.html'), data.index[1]

@app.route("/recommend")
def recommend():
    product = request.args.get('product')
    number = request.args.get('number')
    r = rcmd(product, number)
    #return render_template('recommend.html'),product
    if type(r)==type('string'):
        return render_template('recommend.html',product=product,r=r,t='s')
    else:
        return render_template('recommend.html',product=product,r=r,t='l')



if __name__ == '__main__':
    app.run()