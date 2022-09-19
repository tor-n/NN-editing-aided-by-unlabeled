##AIIKNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random


class UAEditing():
    def __init__(self, clf_names=['knn', 'naivebayes', 'decisiontree'], clf_tune=[False, False, False], clf_params=[{},{},{}], n_iter=4, n_sampled = 0):
        self.clf_names = clf_names
        self.n_iter = n_iter
        self.n_sampled = n_sampled
        # get models
        clf_dict = dict()
        for i, name in enumerate(clf_names):
            d = dict()
            d['clf'] = self.get_model(name)
            d['tune'] = clf_tune[i]
            d['parameters'] = clf_params[i] 
            clf_dict[name] = d
        self.clf_dict = clf_dict
    
    def get_model(self, name):
        if name == 'knn':
            return KNeighborsClassifier()
        elif name == 'naivebayes':
            return GaussianNB()
        elif name == 'decisiontree':
            return DecisionTreeClassifier()
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def add_unlabeled(self, X_unlabeled):
        self.X_unlabeled = X_unlabeled
        self.added_unlabeled_indices = np.array([], dtype='int')
        self.added_unlabeled_labels = np.array([], dtype='int')
        self.new_X = self.X_train
        self.n_sampled = X_unlabeled.shape[0]//4
        return self
    
    def editing(self):
        
        shuffled_indices = np.arange(self.X_unlabeled.shape[0])
        np.random.shuffle(shuffled_indices)
        print(shuffled_indices)
        n_sampled = self.n_sampled
        curr = 0
        for t in range(self.n_iter):
            if t == 0:
                unlabeled_indices = shuffled_indices[curr:curr+n_sampled]
            else:
                unlabeled_indices = np.concatenate((kept_indices, shuffled_indices[curr:curr+n_sampled]))
            curr += n_sampled
            new_indices, kept_indices, new_labels = self.predict_unlabeled(unlabeled_indices)
            n_sampled = self.n_sampled - len(kept_indices)
            print(new_indices)
            print(type(new_indices))
            print(new_labels)
            print(type(new_labels))
            self.added_unlabeled_indices = np.concatenate((self.added_unlabeled_indices,new_indices))
            self.added_unlabeled_labels = np.concatenate((self.added_unlabeled_labels,new_labels))
        
        return self
        

    def predict_unlabeled(self, sampled_indices):
        X_train, y_train, X_unlabeled = self.X_train, self.y_train, self.X_unlabeled
        X_added_unlabeled = X_unlabeled[self.added_unlabeled_indices,:]
        new_X_unlabeled = X_unlabeled[sampled_indices,:]
        new_X_train = np.concatenate((X_train, X_added_unlabeled), axis=0)
        new_y_train = np.concatenate((y_train, self.added_unlabeled_labels))
        clf_dict = self.clf_dict
        
        predicted_labels = np.zeros((len(clf_dict.keys()),new_X_unlabeled.shape[0]))
        
        for i, name in enumerate(self.clf_names):
            clf = clf_dict[name]['clf']
            clf.fit(new_X_train, new_y_train)
            predicted_labels[i,:] = clf.predict(new_X_unlabeled)
            
        equal_bool = np.equal(predicted_labels[0,:], predicted_labels[1,:])& np.equal(predicted_labels[1,:], predicted_labels[2,:])
        equal_indices = np.where(equal_bool)
        unequal_indices = np.where(np.invert(equal_bool))
        
        equal_original_indices = sampled_indices[equal_indices]
        unequal_original_indices = sampled_indices[unequal_indices]
        
        return equal_original_indices, unequal_original_indices, predicted_labels[0, equal_indices[0]]
    
    def augment(self):
        return np.concatenate((self.X_train, self.X_unlabeled[self.added_unlabeled_indices,:])), np.concatenate((self.y_train, self.added_unlabeled_labels))

class AidedRENNfilter():
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def augment(self, X_added, y_added):
        self.X_added = X_added
        self.y_added = y_added
        return self
    
    def screen(self):
        X = np.concatenate((self.X_train, self.X_added), axis=0)
        y = np.concatenate((self.y_train, self.y_added))
        bool_edited = np.full(X.shape[0], True)
        num_edited = 0
        cnt = 0
        while num_edited != np.sum(bool_edited):
            num_edited = np.sum(bool_edited)
            clf = KNeighborsClassifier(n_neighbors=self.k+1)
            clf.fit(X[bool_edited,:], y[bool_edited])
            y_pred = clf.predict(X[bool_edited,:])
            incorrect_bool = y_pred != y[bool_edited]
            edited_indices = np.where(bool_edited)
            print(type(edited_indices))
            bool_edited[edited_indices[0][incorrect_bool]] = False
            cnt += 1
        print(cnt)
        
        return bool_edited
    
    def _get_noisy_indices(self, final_bool):
        clean_indices = np.where(final_bool[:self.X_train.shape[0]])[0]
        noisy_indices = np.where(np.invert(final_bool[:self.X_train.shape[0]]))[0]
        return noisy_indices, clean_indices
            
            
    
  