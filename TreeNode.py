class TreeNode(object):
    def __init__(self, ids = None, children = [], entropy = 0, deep = 0):
        self.ids = ids #номер атрибута в ветке
        self.entropy = entropy #энтропия
        self.deep = deep #как далеко от корня
        self.split_attribute = None #какой атрибут выбран
        self.children = children #список дочерних ветвей
        self.order = None #последовательность атрибута
        self.label = None #метка если это лист дерева

    def set_properties(self, split_atribute, order):
        self.split_attribute = split_atribute
        self.order = order

    def set_label(self, label):
        self.label = label

