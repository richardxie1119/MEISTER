import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import random
import ipywidgets as widgets
from ipywidgets import Box, IntSlider
from sklearn.preprocessing import LabelEncoder

import xgboost
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import nn
import torch

from shap import TreeExplainer, DeepExplainer, KernelExplainer, LinearExplainer

class scMSModel():

	def __init__(self, intens_mtx, metadata=None):
		"""
		"""
		self.model = {}
		self.intens_mtx = intens_mtx
		self.metadata = metadata
		self.names = list(intens_mtx.index)
		self.feature_names = np.array(intens_mtx.columns)

		self.get_models = {'GBT':self.get_GBT_model,'RF':self.get_RF_model,
		'SVM':self.get_SVM_model,'LR':self.get_LR_model,'LDA':self.get_LDA_model,
		'KNN':self.get_KNN_model,'DNN':self.get_DNN_model}

		self.get_explainers = {'GBT':TreeExplainer, 'RF':TreeExplainer,
		'SVM':KernelExplainer, 'LR':LinearExplainer, 'KNN':KernelExplainer,
		'DNN':DeepExplainer}

		self.test_metrics = {}
		self.feature_importance = {}


	def get_labels(self, labels):

		self.label_class = {}

		for label in labels:
			label_encoder = LabelEncoder()
			integer_encoded = label_encoder.fit_transform(self.metadata[label])

			self.label_class[label] = label_encoder.classes_
			self.metadata[label+'_int'] = integer_encoded

			if self.metadata[label].values.dtype == int:
			    self.metadata[label] = self.metadata[label].astype(str)



	def get_kfold_cv(self, k, split_by_name=None):

		self.cv_test = []

		if split_by_name == None:
			#sample_names = self.intens_mtx.index.values
			sample_names_shuffled = shuffle(self.names,random_state=39)
			sample_names_kfold_test = np.array_split(sample_names_shuffled,k)
			self.cv_test = sample_names_kfold_test

		else:
			self.split_names = self.metadata[split_by_name].unique()
			for split_name in self.split_names:
				self.cv_test.append(np.array(self.metadata[self.metadata[split_by_name]==split_name].index))



	def train_models(self, cv, model_names, label_name, shap=False, k=5, feature_names=None,
					 split_by_name=None, learning_rate=0.001, epochs=30, batch_size=32, kwargs=None):

		if cv:
			self.get_kfold_cv(k, split_by_name)

		else:
			k = 1
			test_names = random.sample(self.names, int(len(self.names)/4))
			self.cv_test = [test_names]

		for model_name in model_names:

			k_metric = []
			abs_shap_vals = []
			print('performing {} fold cross validation for {}'.format(len(self.cv_test),model_name))

			if feature_names is not None:
				feature_names = feature_names
			else:
				feature_names = self.feature_names

			for i in range(len(self.cv_test)):

				print('cross validation {}...'.format(i))

				train_names = list(set(self.names) - set(self.cv_test[i]))

				X_train = self.intens_mtx.loc[train_names,feature_names].values.astype(float)
				X_test = self.intens_mtx.loc[self.cv_test[i],feature_names].values.astype(float)
				y_train = self.metadata.loc[train_names][label_name+'_int'].values.astype(int)
				y_test = self.metadata.loc[self.cv_test[i]][label_name+'_int'].values.astype(int)

				if model_name == 'DNN':
					layer_shapes = [X_train.shape[1]]+kwargs['layer_shapes']
					model = self.get_models[model_name](layer_shapes)
					losses, accuracy = self.train_DNN(X_train,y_train,learning_rate, epochs, batch_size)
					p,pred = self.predict_DNN(X_test)

				else:
					model = self.get_models[model_name]()
					model.fit(X_train,y_train)
					pred = model.predict(X_test)

				k_metric.append({'f1':f1_score(pred, y_test,average='micro')})

				if shap:
					shap_val = self.get_shap_vals(X_train, X_test, model_name)
					
					if type(shap_val) is list:
						shap_val = np.concatenate(shap_val)
					abs_shap_vals.append(abs(shap_val))
			if shap:
				self.feature_importance[model_name] = np.concatenate(abs_shap_vals).mean(0)
				self.shap_vals[model_name] = shap_vals

			self.test_metrics[model_name] = k_metric



	def feature_selection(self, n_eliminate, model_name, kwargs, max_feat=None, min_feat=None):

		self.feature_retain = []
		self.feature_retain_metric = []

		self.train_models(**kwargs)
		feature_rank = np.argsort(self.feature_importance[model_name])[::-1]
		self.feature_retain.append(self.feature_names[feature_rank])
		self.feature_retain_metric.append(self.test_metrics.copy())

		feature_nums = np.arange(min_feat,max_feat,n_eliminate)[::-1]

		for i, feat_num in enumerate(feature_nums):

			print('now testing {} features..'.format(feat_num))

			feature_use = self.feature_retain[i][feature_rank[:feat_num]]
			self.train_models(feature_names=feature_use, **kwargs)
			feature_rank = np.argsort(self.feature_importance[model_name])[::-1]
			self.feature_retain.append(feature_use)
			self.feature_retain_metric.append(self.test_metrics.copy())



	def get_shap_vals(self, X, X_shap, model_name):

		explainer = self.get_explainers[model_name]
		model = self.model[model_name]

		if model_name == 'DNN':
			e = explainer(model, Variable(torch.from_numpy(X).float()))
			shap_values = e.shap_values(Variable(torch.from_numpy(X_shap).float()))
		else:
			e = explainer(model, X)
			shap_values = e.shap_values(X_shap)

		return shap_values



	def get_GBT_model(self, learning_rate=0.1,min_split_loss=0,
	              max_depth=6, n_estimators=300,
	              reg_lambda=1,reg_alpha=1,subsample=1):
		"""
		"""

		model = xgboost.XGBClassifier(learning_rate=learning_rate,
		                              min_split_loss=min_split_loss,max_depth=max_depth,
		                              n_estimators=n_estimators,
		                              reg_lambda=reg_lambda,reg_alpha=reg_alpha,
		                              subsample=subsample,use_label_encoder=False,n_jobs=8)
		self.model['GBT'] = model
		return model


	def get_RF_model(self, n_estimators=300, criterion='gini',
	             max_depth=None, min_samples_split=2,
	             min_samples_leaf=1, min_weight_fraction_leaf=0.0,
	             max_features='auto', max_leaf_nodes=None,max_samples=None):

		model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
		                               max_depth=max_depth, min_samples_split=min_samples_split,
		                               min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
		                               max_features=max_features, max_leaf_nodes=max_leaf_nodes,
		                               max_samples=None, random_state=19, n_jobs=8)
		self.model['RF'] = model
		return model


	def get_SVM_model(self, kernel='linear', C=1, max_iter=1e4):

		model = SVC(kernel=kernel,C=C, max_iter=max_iter, random_state=19)

		self.model['SVM'] = model
		return model


	def get_LR_model(self, penalty='l2',tol=0.0001,
	             C=1.0, max_iter=100):


		model = LogisticRegression(penalty=penalty,tol=tol,
		                           C=C,max_iter=max_iter, random_state=19, n_jobs=8)

		self.model['LR'] = model
		return model


	def get_LDA_model(self):

		model = LinearDiscriminantAnalysis()

		self.model['LDA'] = model
		return model


	def get_KNN_model(self, n_neighbors=5, algorithm='auto',
	              leaf_size=30, metric='minkowski'):

		model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm,
		              leaf_size=leaf_size, metric=metric, n_jobs=8)

		self.model['KNN'] = model
		return model



	def get_DNN_model(self, layer_shapes):

		model = Net(layer_shapes=layer_shapes)

		self.model['DNN'] = model
		self.layer_shapes = layer_shapes
		return model



	def train_DNN(self, X_train, y_train, learning_rate=0.001, epochs=30, batch_size=32, verbose=False):

		trainset = dataset(X_train,y_train)
		#DataLoader
		trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

		optimizer = torch.optim.Adam(self.model['DNN'].parameters(),lr=learning_rate)

		if self.layer_shapes[-1] >1:
			loss_fn = nn.CrossEntropyLoss()
		else:
			loss_fn = nn.BCELoss()

		losses = []
		accuracy = []
		for i in range(epochs):
			for j,(x_batch,y_batch) in enumerate(trainloader):

				#calculate output
				output = self.model['DNN'](x_batch)

				#calculate loss
				if self.layer_shapes[-1] >1:
					loss = loss_fn(output, Variable(y_batch).long())
				else:
					loss = loss_fn(output,y_batch.reshape(-1,1))

				#accuracy
				predicted = self.model['DNN'](torch.tensor(X_train, dtype=torch.float32))
				if self.layer_shapes[-1] >1:
					__, predicted_labels = torch.max(predicted, dim = 1)
				else:
					predicted_labels = predicted.reshape(-1).round()

				acc = (predicted_labels.detach().numpy() == y_train).mean()
				#backprop
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			losses.append(loss)
			accuracy.append(acc)

			if verbose:
				if i % 5 ==0:
					print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))

		return losses, accuracy


	def predict_DNN(self, X_test):

		predicted = self.model['DNN'](torch.tensor(X_test, dtype=torch.float32))

		if self.layer_shapes[-1] >1:
			__, predicted_labels = torch.max(predicted, dim = 1)
		else:
			predicted_labels = predicted.reshape(-1).round()

		return predicted.detach().numpy(), predicted_labels.detach().numpy()



class dataset(Dataset):

	def __init__(self,x,y):
	    self.x = torch.tensor(x,dtype=torch.float32)
	    self.y = torch.tensor(y,dtype=torch.float32)
	    self.length = self.x.shape[0]

	def __getitem__(self,idx):
	    return self.x[idx],self.y[idx]

	def __len__(self):
	    return self.length



class Net(nn.Module):

	def __init__(self,layer_shapes):

		super(Net,self).__init__()

		self.layers = nn.ModuleList()
		self.layer_shapes = layer_shapes

		for i in range(len(layer_shapes)-1):
			self.layers.append(nn.Linear(layer_shapes[i],layer_shapes[i+1]))

	def forward(self,x):
	    for i in range(len(self.layers)-1):
	    	x = torch.relu(self.layers[i](x))

	    if self.layer_shapes[-1] >1:
	    	x = self.layers[-1](x)
	    else:
	    	x = torch.sigmoid(self.layers[-1](x))

	    return x


from sklearn.metrics import classification_report, confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure(figsize=(7,7))
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
