#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This Code is test for query cluster research based on term.
The development environment is Python 2.7 64bit.
And Package include:
    gensim,depend on numpy and scipy
    scikit-learn,depend on numpy and scipy
    Matplotlib
If you want to run this code,you should install these package above.

Firstly,we use Tfidf model in gensim to transform a string words to integer value vector.
And we use lda or lsi model in gensim to transform integer value vector to real value vector.
Then,we use some clustering algorithm in scikit-learn to cluster the real value vector.
Lastly,we plot the cluster result in 3-D figure.
"""

import gensim,logging
from gensim import corpora,models,similarities
from gensim.models import *
from sklearn import *
from sklearn.cluster import *
from sklearn.metrics import *
from sklearn.decomposition import *
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from time import time
import linecache


#we will filter the word in stoplist for corpora
stoplist = set('for a of the and to in'.split())

#read line from file and filter the word in stoplist
class MyCorpus(object):
    def __init__(self,file):
        self.file = file
    def __iter__(self):
        for line in open(self.file):
            text = [word for word in line.lower().split() if word not in stoplist]
            yield text


#Train the vector model from corpora
def Train_Vector_Model(file,query):
    logger.info('Train vector model...')
    filename = file.split('.')[0]
    #input corpora text file and get a string words
    Corp = MyCorpus(file)

    dictionary = corpora.Dictionary(Corp)

    #save corpora to disk in .dict format
    dictionary.save(filename + '.dict')

    #transform a string words to integer value vector and save to disk in .mm format
    corpus = [dictionary.doc2bow(text) for text in Corp]
    corpora.MmCorpus.serialize(filename + '.mm', corpus)


    t0 = time()
    #load the .dict file and .mm file from disk
    dictionary = corpora.Dictionary.load(filename + '.dict')
    corpus = corpora.MmCorpus(filename + '.mm')

    #transform integer value vector to real value vector through Tfidf model 
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]


    #transform real value vector to a normalized vector through Lsi or Lda model,
    #and num_topics is the dimension of normalized vector

    model = lsimodel.LsiModel(corpus_tfidf,id2word = dictionary,num_topics = 30)
    #model = ldamodel.LdaModel(corpus_tfidf,id2word = dictionary,num_topics = 30)

    model.save(filename + '.lsi')
    #model.save(filename + '.lda')


    #model = lsimodel.LsiModel.load(filename + '.lsi')
    #model = ldamodel.LdaModel.load(filename + '.lda')

    corpus_model = model[corpus_tfidf]

    if query != None:
        index = similarities.MatrixSimilarity(model[corpus])
        query_list = [word for word in query.lower().split() if word not in stoplist]
        query_bow = dictionary.doc2bow(query_list)
        query_model = model[query_bow]
        sims = index[query_model]
        sort_sims = sorted(enumerate(sims),key=lambda item:-item[1])
    else:
        sort_sims = None

    vector_space_model = zeros((corpus.num_docs,model.num_topics))
    for index1,element in enumerate(corpus_model):
        for index2,ele in element:
            vector_space_model[index1][index2] = ele
    Elapsed_Time1 = time() - t0
    logger.info("Train vector model elapsed time:  %.3fs" %Elapsed_Time1)
    return vector_space_model,sort_sims

#cluster in the vector model
def Cluster_Process(vector_space_model):
    #Start cluster
    t0 = time()


    logger.info("Compute Kmeans clustering...")
    n_clusters = 4
    k_means = KMeans(n_clusters).fit(vector_space_model)
    labels = k_means.labels_
    n_clusters_ = max(labels) + 1


    '''
    logger.info("Compute MeanShift clustering...")
    bandwidth = estimate_bandwidth(vector_space_model,quantile=0.4)

    ms = MeanShift(bandwidth = bandwidth,bin_seeding=False).fit(vector_space_model)
    labels = ms.labels_
    clusters_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    '''


    '''
    logger.info("Compute DBSCAN clustering...")
    db = DBSCAN(eps = 0.08,min_samples = 100).fit(vector_space_model)
    core_sample_mask = np.zeros_like(db.labels_,dtype = bool)
    core_sample_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    '''

    logger.info("Cluster elapsed time:  %.3fs" % (time() - t0))
    logger.info("Estimated number of cluster: %d",n_clusters)
    logger.info("labels = %s"%labels)
    dict_labels = {}
    for index,element in enumerate(labels):
        if element in dict_labels:
            dict_labels[element] += 1
        else:
            dict_labels[element] = 1
    logger.info("dict_labels = %s"%dict_labels)
    #logger.info("silhouette coefficient = ",metrics.silhouette_score(vector_space_model,labels,metric = 'euclidean'))
    return labels


#Store cluster result to disk
def Store_Cluster_Result(file,labels):
    logger.info('Store Cluster Result...')

    filename = file.split('.')[0]

    t0 = time()
    for i in np.unique(labels):
        f_output = open(filename + '_cluster_%d.txt'%i,'w')
        for index,element in enumerate(labels):
            if element == i:
                f_output.write(linecache.getline(file,index+1))
        f_output.close()
    logger.info("Store Cluster Result elapsed time:  %.3fs" % (time() - t0))

#figure out the cluster result
def Plot_Cluster_Result(vector_space_model,labels):
    #plot cluster label in 2-D
    logger.info("Plot cluster result")

    #plot cluster result in 3-D
    pca = PCA(n_components = 3)
    X = pca.fit(vector_space_model).transform(vector_space_model)
    fig2 = plt.figure('cluster result')
    ax = p3.Axes3D(fig2)
    ax.view_init(7,-80)
    for l in np.unique(labels):
        ax.plot3D(X[labels == l,0],X[labels == l,1],X[labels == l,2],'o',
              color = plt.cm.jet(np.float(l)/np.max(labels + 1)))
    plt.show()

#main function
def main():
    #if you want to get the similarity between a specific string and s set of string
    file = 'eng_query_result.txt'
    query = None

    vector_space_model,sort_sims = Train_Vector_Model(file,query)

    labels = Cluster_Process(vector_space_model)

    Store_Cluster_Result(file,labels)

    Plot_Cluster_Result(vector_space_model, labels)

    if sort_sims != None:
        logger.info('sort_sims:%s'%sort_sims[:100])

if __name__ == "__main__":
    print(__doc__)
    logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s',level = logging.INFO)
    logger = logging.getLogger('word2vec')
    main()