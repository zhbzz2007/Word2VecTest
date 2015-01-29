#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim,logging
from gensim import corpora,models,similarities
from gensim.models import lsimodel
from sklearn import *
from sklearn.cluster import *
from sklearn.metrics import *
from sklearn.decomposition import *
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from time import time

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s',level = logging.INFO)

'''
stoplist = set('for a of the and to in'.split())
class MyCorpus(object):
    def __init__(self,filename):
        self.filename = filename
    def __iter__(self):
        for line in open(self.filename):
            text = [word for word in line.lower().split() if word not in stoplist]
            yield text

Corp = MyCorpus('mycorpus.txt')

dictionary = corpora.Dictionary(Corp)

dictionary.save('mycorpus.dict')

print 'dictionary.token2id:\n',dictionary.token2id

corpus = [dictionary.doc2bow(text) for text in Corp]
corpora.MmCorpus.serialize('mycorpus.mm', corpus)

#print '\ncorpus:\n',corpus
f = open('result.txt','w')
for element in corpus:
    f.write(str(element)+'\n')
f.close()

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec) # the word "interaction" does not appear in the dictionary and is ignored

print 'end'
'''

#transform bag-of-word to integer truth vector space
dictionary = corpora.Dictionary.load('mycorpus.dict')
corpus = corpora.MmCorpus('mycorpus.mm')

tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]


'''
#transform integer truth vector space to real true vector space through LsiModel
#and save file to disk and load this this file next time 
model = lsimodel.LsiModel(corpus_tfidf,id2word=dictionary,num_topics = 5)
print model
model.save('mycorpus.lsi')
'''

model = lsimodel.LsiModel.load('mycorpus.lsi')


corpus_model = model[corpus_tfidf]

vector_space_model = zeros((corpus.num_docs,model.num_topics))
for index1,element in enumerate(corpus_model):
    for index2,ele in element:
        vector_space_model[index1][index2] = ele
print 'vector_space_model:\n',vector_space_model


vector_space_model = datasets.load_iris().data

#---------------------------------------------------------------------
#Firstly,use AffinityPropagation,MeanShift,DBSCAN to cluster and choose the best cluster by max the silhouette index function
#and use the cluster number as input of K-Means and Spectral cluster to cluster.
silhouette_score_max = -1
n_clusters = -1
labels = []
s = {AffinityPropagation:0,MeanShift:1,DBSCAN:2}
cluster_algorithm = iter(s)
for index in cluster_algorithm:
    if index == AffinityPropagation:
        print("Compute structured AffinityPropagation clustering...")
        t0 = time()
        af = index(damping=0.98, max_iter=400, convergence_iter=10, copy=True,
                   preference=None, affinity='euclidean', verbose=False).fit(vector_space_model)
        elapsed_time = time() - t0
        print("Elapsed time: %.3fs" % elapsed_time)
        labels_ = af.labels_
        n_clusters_ = len(af.cluster_centers_indices_)
    elif index == MeanShift:
        print("Compute structured MeanShift clustering...")
        bandwidth = estimate_bandwidth(vector_space_model,quantile=0.5)
        t0 = time()
        ms = index(bandwidth = bandwidth,bin_seeding=True).fit(vector_space_model)
        elapsed_time = time() - t0
        print("Elapsed time: %.3fs" % elapsed_time)
        labels_ = ms.labels_
        n_clusters_ = len(np.unique(labels_))
    else:
        print("Compute structured DBSCAN clustering...")
        t0 = time()
        db = index(eps = 0.4,min_samples = 10).fit(vector_space_model)
        elapsed_time = time() - t0
        print("Elapsed time: %.3fs" % elapsed_time)
        core_sample_mask = np.zeros_like(db.labels_,dtype = bool)
        core_sample_mask[db.core_sample_indices_] = True
        labels_ = db.labels_
        n_clusters_ = len(set(labels_)) - (1 if -1 in labels_ else 0)
    silhouette_score_ = metrics.silhouette_score(vector_space_model,labels_,metric = 'euclidean')
    if silhouette_score_ > silhouette_score_max:
        silhouette_score_max = silhouette_score_
        labels = labels_
        n_clusters = n_clusters_

print 'Estimated number of cluster:',n_clusters
print("Compute structured K-Means clustering...")
t0 = time()
k_means = KMeans(n_clusters).fit(vector_space_model)
#k_means = MiniBatchKMeans(n_clusters,batch_size = 10).fit(vector_space_model)
elapsed_time = time() - t0
print("Elapsed time: %.3fs" % elapsed_time)
labels = k_means.labels_
print 'labels = ',labels
print 'silhouette coefficient = ',metrics.silhouette_score(vector_space_model,labels,metric = 'euclidean')


print("Plot Cluster Label...")
dict_labels = {}
plt.suptitle("Cluster Label", size = 15)
plt.grid(b = None, which = 'major', axis='both')
for index,element in enumerate(labels):
    if element in dict_labels:
        dict_labels[element] += 1
    else:
        dict_labels[element] = 1
    plt.plot(index,element,'o',color = plt.cm.jet(np.float(element + 1)/np.max(labels + 1)))
print 'dict_labels = ',dict_labels


#---------------------------------------------------
#plot result in 3-D
print("Plot Cluster Distribution...")
print("Dimensionality reduction by PCA...")
pca = PCA(n_components = 3)
X = pca.fit(vector_space_model).transform(vector_space_model)
fig2 = plt.figure('figure2')
fig2.suptitle('Cluster Distribution',size = 15)
ax = p3.Axes3D(fig2)
ax.view_init(7,-80)
for l in np.unique(labels):
    ax.plot3D(X[labels == l,0],X[labels == l,1],X[labels == l,2],'o',
        color = plt.cm.jet(np.float(l)/np.max(labels + 1)))
plt.show()
