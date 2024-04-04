import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.model_selection import cross_val_score
from scipy.sparse import lil_matrix
import numpy as np
import json
from time import time
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro


def eva(y_true, y_pred, epoch='end'):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print(epoch, ':acc {:.2f}'.format(100*acc), ', nmi {:.2f}'.format(100*nmi), ', ari {:.2f}'.format(100*ari),
            ', f1 {:.2f}'.format(100*f1))
    

def format_data_for_display(emb_file, i2l_file):
	i2l = dict()
	with open(i2l_file, 'r') as r:
		r.readline()
		for line in r:
			parts = line.strip().split(',')
			n_id, l_id = int(parts[0]), int(parts[1])
			i2l[n_id] = l_id
	
	i2e = dict()
	with open(emb_file, 'r') as r:
		r.readline()
		for line in r:
			embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
			node_id = embeds[0]
			if node_id in i2l:
				i2e[node_id] = embeds[1:]
	
	i2l_list = sorted(i2l.items(), key=lambda x:x[0])
	
	X = []
	Y = []
	for (id, label) in i2l_list:
		X.append(i2e[id])
		Y.append(label)
		
	return X,Y

def getdata(oremb, orlabel):
    # print(target)
    target = [0, 1]
    emb = []
    label = []
    nodechoice = dict()
    index = 0
    for i in orlabel:
        if i in target:
            if i not in nodechoice:
                nodechoice[i]=set()
            nodechoice[i].add(index)
        index=index+1
    for i in target:
        print(str(i)+' '+str(len(nodechoice[i])))
        if i == 1:
            # temp = random.choices(list(nodechoice[i]), k=len(nodechoice[i]))
            temp = random.choices(list(nodechoice[i]), k=min(len(nodechoice[i]), 2))
        else:
            temp = random.choices(list(nodechoice[i]), k=min(len(nodechoice[1]), 2))
        for index in temp:
            emb.append(oremb[index])
            label.append(orlabel[index])
    
    return emb, label

def run(code):
    oremb, orlabel = format_data_for_display('./node_emb/tgn_bitalpha.emb', './node_emb/label1.csv')
    # if code=='gca':
    #     oremb, orlabel = format_data_for_display('./node_emb/bitalpha.emb', '../../dataset/ml_bitalpha.csv')
    # else :
    #     oremb, orlabel = format_data_for_display('./node_emb/{}_mooc.emb'.format(code), '../../dataset/ml_mooc.csv')
    # emb, label = getdata(oremb, orlabel)
    emb, label = oremb, orlabel
	#print(label)
    kmeans = KMeans(n_clusters=7, n_init=20)
    result = kmeans.fit_predict(emb)
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    # tstart = time()
    # result = tsne.fit_transform(emb)
    eva(label, result, code)

code = 'gca'
run(code)
# python train_supervised.py -d mooc --use_memory --prefix tgn-attn-mooc --n_runs 1