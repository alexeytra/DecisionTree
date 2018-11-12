from TreeNode import TreeNode
import numpy as np
import math


def entropy(freq):
    # remove prob 0
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    en = np.sum(-prob_0*np.log2(prob_0))
    return en

class DecisionTree(object):
    def __init__(self, max_deep = 10, min_samples_split = 2, min_gain = 1e-4):
        self.root = None
        self.max_depth = max_deep
        self.min_samples_split = min_samples_split
        self.Ntrain = 0
        self.min_gain = min_gain

    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data
        self.attributes = list(data)
        self.target = target
        self.labels = target.unique()

        ids = range(self.Ntrain)

        self.root = TreeNode(ids=ids, entropy=self._entropy(ids), deep=0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.deep < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children:  # лист
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)

    def _set_label(self, node):

        target_ids = [i + 1 for i in node.ids]
        node.set_label(self.target[target_ids].mode()[0])

    def _split(self, node):
        ids = node.ids
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            if len(values) == 1: continue  # entropy = 0
            splits = []
            for val in values:
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id - 1 for sub_id in sub_ids])

            if min(map(len, splits)) < self.min_samples_split: continue
            # ig
            HxS = 0
            for split in splits:
                HxS += len(split) * self._entropy(split) / len(ids)
            gain = node.entropy - HxS
            if gain < self.min_gain: continue  # stop if small gain
            if gain > best_gain:
                best_gain = gain
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids=split, entropy=self._entropy(split), deep=node.deep + 1) for split in best_splits]
        return child_nodes

    def predict(self, new_data):

        npoints = new_data.count()[0]
        labels = [None] * npoints
        for n in range(npoints):
            x = new_data.iloc[n, :]  # one point

            node = self.root
            while node.children:
                node = node.children[node.order.index(x[node.split_attribute])]
            labels[n] = node.label

        return labels

    def _entropy(self, ids):

        if len(ids) == 0: return 0
        ids = [i + 1 for i in ids]
        freq = np.array(self.target[ids].value_counts())
        return entropy(freq)