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


stoplist = set('for a of the and to in'.split())
class MyCorpus(object):
    def __init__(self,filename):
        self.filename = filename
    def __iter__(self):
        for line in open(self.filename):
            text = [word for word in line.lower().split() if word not in stoplist]
            yield text

Corp = MyCorpus('eng_query_ansi_result.txt')

dictionary = corpora.Dictionary(Corp)

dictionary.save('result.dict')

print 'dictionary.token2id:\n',dictionary.token2id

corpus = [dictionary.doc2bow(text) for text in Corp]
corpora.MmCorpus.serialize('result.mm', corpus)

'''
f = open('result.txt','w')
for element in corpus:
    f.write(str(element)+'\n')
f.close()
print 'end'
'''

dictionary = corpora.Dictionary.load('result.dict')
corpus = corpora.MmCorpus('result.mm')

tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]


model = lsimodel.LsiModel(corpus_tfidf,id2word = dictionary,num_topics = 20)
print model
model.save('result.lsi')
'''
model = lsimodel.LsiModel.load('result.lsi')
'''

f = open('result.txt','w')
corpus_model = model[corpus_tfidf]
vector_space_model = zeros((corpus.num_docs,model.num_topics))
for index1,element in enumerate(corpus_model):
    for index2,ele in element:
        vector_space_model[index1][index2] = ele
        f.write(str(ele)+'\t')
    f.write('\n')
f.close()


t0 = time()

n_clusters = 5
k_means = cluster.KMeans(n_clusters).fit(vector_space_model)
labels = k_means.labels_
n_clusters_ = max(labels) + 1




'''
#---------------------------------------------------------------------
#测试Ap
print("Compute structured AffinityPropagation clustering...")
af = AffinityPropagation(damping=0.8, max_iter=20, convergence_iter=10, copy=True,
     preference=None, affinity='euclidean', verbose=False).fit(vector_space_model)

cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters = len(cluster_centers_indices)
'''


'''
#---------------------------------------------------------------------
#测试MeanShift
print("Compute structured MeanShift clustering...")
bandwidth = estimate_bandwidth(vector_space_model,quantile=0.5)

ms = MeanShift(bandwidth = bandwidth,bin_seeding=True).fit(vector_space_model)
labels = ms.labels_
clusters_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters = len(labels_unique)
'''


'''
print("Compute structured DBSCAN clustering...")
db = DBSCAN(eps = 0.08,min_samples = 100).fit(vector_space_model)
core_sample_mask = np.zeros_like(db.labels_,dtype = bool)
core_sample_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
'''


print("Elapsed time: %.3fs" % (time() - t0))
print 'Estimated number of cluster:',n_clusters
print 'labels = ',labels
dict_labels = {}
for index,element in enumerate(labels):
    if element in dict_labels:
        dict_labels[element] += 1
    else:
        dict_labels[element] = 1
print 'dict_labels = ',dict_labels
#print 'silhouette coefficient = ',metrics.silhouette_score(vector_space_model,labels,metric = 'euclidean')


plt.grid(b = None, which = 'major', axis='both')
plt.plot(range(vector_space_model.shape[0]),labels,'ro')
#plt.show()

#---------------------------------------------------
#plot result in 3-D
pca = PCA(n_components = 3)
vector_space_model_pca = pca.fit(vector_space_model).transform(vector_space_model)
X = vector_space_model_pca
fig2 = plt.figure('figure2')
ax = p3.Axes3D(fig2)
ax.view_init(7,-80)
for l in np.unique(labels):
    ax.plot3D(X[labels == l,0],X[labels == l,1],X[labels == l,2],'o',
        color = plt.cm.jet(np.float(l)/np.max(labels + 1)))
plt.show()
