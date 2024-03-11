from collections import defaultdict, Counter
import numpy as np

class StrategyTree(object):
    def __init__(self,classes,features,max_depth=10,min_samples_split=10,type='entropy'):
        self.classes = classes
        self.features = features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.type =type
        self.root=None
        self.tree=defaultdict(list)

    def get_params(self, deep):
        return {'classes': self.classes, 'features': self.features,
                'max_depth': self.max_depth, 'min_samples_split': self.min_samples_split,
                '__impurity_t': self.type}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def traversal(self, node, sample_col):
        assert len(sample_col) == len(self.features)
        if type(node) is not tuple:
            return node
        determin_feature=sample_col[node[0]]
        if determin_feature in node[1]:
           return  self.traversal(node[1][determin_feature], sample_col)
        return node[-1]


    def predict(self, samples_matrix):
        assert len(samples_matrix.shape) == 1 or len(samples_matrix.shape) == 2
        if samples_matrix.shape == 1:
            return self.traversal(self.root)
        return np.array([self.traversal(self.root, col) for col in samples_matrix])


    def __impurity(self, labels):
        labels_accumulator,cap=Counter(labels),float(len(labels))
        label_index=defaultdict(list)
        for index,label in enumerate(labels):
            label_index[label].append(index)

        probabilities=[labels_accumulator[k]/cap for k in labels_accumulator]
        if self.type =='gini':
            return 1-sum([p*p for p in probabilities])
        return -sum([p*np.log2(p) for p in probabilities])



    def __gain(self,feature_vertex,labels)->tuple:
        label_entropy=self.__impurity(labels)
        feature_col_map=defaultdict(list)

        for _index,feature in enumerate(feature_vertex):
            feature_col_map[feature].append(_index)


        feature_label_entropy=0
        label_len=len(labels)
        for v in feature_col_map:
            f_l=labels[feature_col_map[v]]
            feature_label_entropy+=self.__impurity(f_l)*len(f_l)/label_len
        feature_entropy=self.__impurity(feature_vertex)
        r=(label_entropy-feature_label_entropy)/(feature_entropy if feature_entropy!=0 else 1)

        return [feature_col_map,r]

    def expand_node(self,matrix,labels,depth,used_features):
        #if there is only one value to labels row
        if len(set(labels))==0:
            return labels[0]
        most_frequent_label=Counter(labels).most_common(1)[0][0]
        if depth>=self.max_depth or len(labels) < self.min_samples_split:
            return most_frequent_label

        best_feature_index,max_score,f_index=-1,-1,None
        for feature_row_index in range(len(self.features)):
            if feature_row_index in used_features:
                continue
            cur_feature_col_map,cur_feature_score=self.__gain(matrix[:,feature_row_index],labels)

            if best_feature_index<0 or cur_feature_score > max_score:
                  best_feature_index,max_score,f_index=feature_row_index,cur_feature_score,cur_feature_col_map


        if best_feature_index < 0:
            return most_frequent_label

        to_next_node={}
        new_used_feature=used_features + [best_feature_index]
        for _f in f_index:
            next_cor_index=f_index[_f]
            to_next_node[_f]=self.expand_node(matrix[next_cor_index,:],labels[next_cor_index],depth+1,new_used_feature)



        self.tree[depth].append(self.features[best_feature_index])
        return (best_feature_index,to_next_node,most_frequent_label)


    def fit(self,train_matrix,label):
        assert len(self.features) == len(train_matrix[0])
        self.root=self.expand_node(train_matrix,label,depth=1,used_features=[])





