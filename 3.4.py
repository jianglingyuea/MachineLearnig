#比较十折交叉验证和留一法
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.datasets import load_wine
# 载入wine数据
dataset = load_wine()
#10次10折交叉验证法生成训练集和测试集
def kfolds(n):
    k = 0
    truth = 0
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    kf = model_selection.KFold(n_splits=n, random_state=None, shuffle=True)
    for x_train_index, x_test_index in kf.split(dataset.data):
        x_train.append(dataset.data[x_train_index])
        y_train.append(dataset.target[x_train_index])
        x_test.append(dataset.data[x_test_index])
        y_test.append(dataset.target[x_test_index])
    while k < n:
        # 用对率回归进行训练，拟合数据
        log_model = LogisticRegression(solver='liblinear', multi_class='auto')
        log_model.fit(x_train[k], y_train[k])
        # 用训练好的模型预测
        y_pred = log_model.predict(x_test[k])
        for i in range(len(x_test[k])):         #这里和留一法不同，是因为10折交叉验证的验证集是len(dataset.target)/10，验证集的预测集也是，都是一个列表，是一串数字，而留一法是一个数字
            if y_pred[i] == y_test[k][i]:
                truth += 1
        k += 1
    # 计算精度
    accuracy = truth/len(dataset.data)  #accuracy = truth/len(dataset.target)
    return(accuracy)
acc = kfolds(10)
print("10折交叉验证对率回归的精度是：", acc)



#留一法
acc = kfolds(len(dataset.data))
print("留一法对率回归的精度是：", acc)