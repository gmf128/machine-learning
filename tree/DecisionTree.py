import math

default_class = 0
IV = [0, 0, 0]
class DecisionNode:
    def __init__(self, col=-1, results=None, children=None):
        self.col = col          # 切分属性的列索引
        self.results = results  # 叶节点中的分类结果
        self.children = children  # 子节点

# 对数据集按照属性列col进行分类
def divideSet(rows, col, value):
    splittingFunction = None
    if col == 1:  # 处理数值型数据
        splittingFunction = lambda row: row[col] >= value
    else:  # 处理字符串型数据
        splittingFunction = lambda row: row[col] == value
    # 将数据集划分成两个集合，并返回
    set1 = [row for row in rows if splittingFunction(row)]
    set2 = [row for row in rows if not splittingFunction(row)]
    return (set1, set2)

# 统计每个分类结果出现的次数
def uniqueCounts(rows, col=-1):
    results = {}
    for row in rows:
        r = row[col]
        if r not in results.keys():
            results[r] = 0
        results[r] += 1
    return results

def update_default(rows):
    """更新default_class的值"""
    results = uniqueCounts(rows)
    maxnum = 0
    maxindex = 0
    for i in results.keys():
        num = results[i]
        if num >= maxnum:
            maxindex = i
            maxnum = num
    global default_class
    global IV
    default_class = maxindex
    """更新增益律的值"""
    log2 = lambda x: math.log(x) / math.log(2)
    tmp = uniqueCounts(rows, 0)
    for key in tmp.keys():
        p = tmp[key]/len(rows)
        IV[0] += -p * log2(p)
    tmp = uniqueCounts(rows, 2)
    for key in tmp.keys():
        p = tmp[key] / len(rows)
        IV[2] += -p * log2(p)

# 计算熵
def entropy(rows):
    log2 = lambda x: math.log(x) / math.log(2)
    results = uniqueCounts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p * log2(p)
    return ent

# 选择最佳切分属性
def buildDecisionTree(rows, scoref=entropy, groupedCol=[-1]):
    if len(rows) == 0:
        return DecisionNode()

    currentScore = scoref(rows)

    # 更新default_class的值
    if currentScore == 0:
      result = [r for r in uniqueCounts(rows).keys()]
      return DecisionNode(results=result[0])

    if 0 in groupedCol and 1 in groupedCol and 2 in groupedCol:
        result = [r for r in uniqueCounts(rows).keys()]
        return DecisionNode(results=result[0])

    bestGain = 0.0
    bestAttribute = None
    bestSets = None

    columnCount = len(rows[0]) - 1
    for col in range(columnCount):
        if col in groupedCol:
            continue
        columnValues = list(set([row[col] for row in rows]))
        # col 1 :视作连续变量处理
        if col == 1:
            columnValues = [100, 50, 20, 10, 5, 0]
        # 对于每个属性，计算切分数据集之后的信息增益
        sum = 0
        sets = rows
        childeren = {}
        for value in columnValues:
            (set1, set2) = divideSet(sets, col, value)
            # 信息增益
            p = float(len(set1)) / len(rows)
            sum += p * scoref(set1)
            childeren[value] = set1
            sets = set2
        # 信息增益
        gain = currentScore - sum
        if col == 0:
            gain = gain/IV[0]
        elif col == 2:
            gain = gain/IV[2]
        if gain > bestGain:
            bestGain = gain
            bestAttribute = col
            bestSets = childeren
    # 创建子分支
    if bestGain > 0:
        childeren = {}
        groupedCol.append(bestAttribute)
        for index in bestSets.keys():
            childeren[index] = buildDecisionTree(bestSets[index], groupedCol=groupedCol)
        return DecisionNode(col=bestAttribute, children=childeren)
    else:
        # 叶节点: 即bestGain=0,即这块数据集在属性上取值全部相同或为空集。或者说明任何划分会导致信息熵增加，说明无需划分
        results = uniqueCounts(rows)
        maxnum = 0
        maxindex = 0
        for i in results.keys():
            num = results[i]
            if num >= maxnum:
                maxindex = i
                maxnum = num
        return DecisionNode(results=maxindex)

def classify(observation, tree):
    if tree.results != None: # 叶节点
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if tree.col == 1:
            if v >= 100:
                branch = tree.children[100]
            elif v >= 50:
                branch = tree.children[50]
            elif v >= 20:
                branch = tree.children[20]
            elif v >= 10:
                branch = tree.children[10]
            elif v >= 5:
                branch = tree.children[5]
            elif v >= 0:
                branch = tree.children[0]
        else:
            try:
                branch = tree.children[v]
            except:
                return default_class
        return classify(observation, branch)


# import math
#
# class DecisionNode:
#     def __init__(self, col=-1, value=None, results=None, leftChildren=None, rightChildren=None):
#         self.col = col          # 切分属性的列索引
#         self.value = value      # 切分属性的取值
#         self.results = results  # 叶节点中的分类结果
#         self.leftChildren = leftChildren            # 左子树
#         self.rightChildren = rightChildren            # 右子树
#
# # 对数据集按照属性列col进行分类
# def divideSet(rows, col, value):
#     splittingFunction = None
#     if col == 1:  # 处理数值型数据
#         splittingFunction = lambda row: row[col] >= value
#     else:  # 处理字符串型数据
#         splittingFunction = lambda row: row[col] == value
#     # 将数据集划分成两个集合，并返回
#     set1 = [row for row in rows if splittingFunction(row)]
#     set2 = [row for row in rows if not splittingFunction(row)]
#     return (set1, set2)
#
# # 统计每个分类结果出现的次数
# def uniqueCounts(rows):
#     results = {}
#     for row in rows:
#         r = row[-1]
#         if r not in results:
#             results[r] = 0
#         results[r] += 1
#     return results
#
#
# # 计算熵
# def entropy(rows):
#     log2 = lambda x: math.log(x) / math.log(2)
#     results = uniqueCounts(rows)
#     ent = 0.0
#     for r in results.keys():
#         p = float(results[r]) / len(rows)
#         ent -= p * log2(p)
#     return ent
#
# # 选择最佳切分属性
# def buildDecisionTree(rows, scoref=entropy):
#     if len(rows) == 0:
#         return DecisionNode()
#
#     currentScore = scoref(rows)
#
#     if currentScore == 0:
#       result = [r for r in uniqueCounts(rows).keys()]
#       return DecisionNode(results=result[0])
#
#     bestGain = 0.0
#     bestAttribute = None
#     bestSets = None
#
#     columnCount = len(rows[0]) - 1
#     for col in range(columnCount):
#         columnValues = list(set([row[col] for row in rows]))
#         # col 1 :视作连续变量处理
#         if col == 1:
#             columnValues = [100, 50, 20, 10, 5, 0]
#         # 对于每个属性，计算切分数据集之后的信息增益
#         sum = 0
#         sets = rows
#         childeren = {}
#         for value in columnValues:
#             (set1, set2) = divideSet(sets, col, value)
#             # 信息增益
#             p = float(len(set1)) / len(rows)
#             sum += p * scoref(set1)
#             childeren[value] = set1
#             sets = set1
#         # 信息增益
#         gain = currentScore - sum
#         if gain > bestGain:
#             bestGain = gain
#             bestAttribute = col
#             bestSets = childeren
#     # 创建子分支
#     if bestGain > 0:
#         trueBranch = buildDecisionTree(bestSets[0])
#         falseBranch = buildDecisionTree(bestSets[1])
#         return DecisionNode(col=bestAttribute[0], value=bestAttribute[1], leftChildren=trueBranch, rightChildren=falseBranch)
#     else:
#         # 叶节点: 即bestGain=0,即这块数据集在属性上取值全部相同或为空集
#         results = uniqueCounts(rows)
#         maxnum = 0
#         maxindex = 0
#         for i in results.keys():
#             num = results[i]
#             if num >= maxnum:
#                 maxindex = i
#                 maxnum = num
#         return DecisionNode(results=maxindex)
#
# def classify(observation, tree):
#     if tree.results != None: # 叶节点
#         return tree.results
#     else:
#         v = observation[tree.col]
#         branch = None
#         if tree.col == 1:
#             if v >= tree.value:
#                 branch = tree.leftChildren
#             else:
#                 branch = tree.rightChildren
#         else:
#             if v == tree.value:
#                 branch = tree.leftChildren
#             else:
#                 branch = tree.rightChildren
#         return classify(observation, branch)