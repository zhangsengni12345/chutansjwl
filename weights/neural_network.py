import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from PIL import Image

class neuralNetwork :

    # 用于神经网络初始化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 输入层节点数
        self.inodes = inputnodes
        # 隐层节点数
        self.hnodes = hiddennodes
        # 输出层节点数
        self.onodes = outputnodes
        # 学习率
        self.lr = learningrate

        # 初始化输入层与隐层之间的权重
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 初始化隐层与输出层之间的权重
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数（S函数）
        self.activation_function = lambda x: scipy.special.expit(x)
        # 逆激活函数
        self.iactivation_function = lambda y: scipy.special.logit(y)

    # 设置权重
    def setweights(self, wih, who):
        self.wih = wih
        self.who = who

    # 神经网络学习训练
    def train(self, inputs_list, targets_list):
        # 将输入数据转化成二维矩阵
        inputs = np.array(inputs_list, ndmin=2).T
        # 将输入标签转化成二维矩阵
        targets = np.array(targets_list, ndmin=2).T

        # 计算隐层的输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐层的输出
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的输出
        final_outputs = self.activation_function(final_inputs)

        # 计算输出层误差
        output_errors = targets - final_outputs
        # 计算隐层误差
        hidden_errors = np.dot(self.who.T, output_errors)

        # 更新隐层与输出层之间的权重
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # 更新隐层与输出层之间的权重
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    # 根据标签逆向生成图片
    def makeinput(self, outputs_list):
        # 将标记数据（输出层的输出）转化成二维矩阵
        final_outputs = np.array(outputs_list, ndmin=2).T
        # 计算输出层的输入
        final_inputs = self.iactivation_function(final_outputs)

        # 计算隐层的输出
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # 处理下限使得下限为0
        hidden_outputs -= np.min(hidden_outputs)
        # 处理上限使得上限为1
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        # 计算隐层的输入
        hidden_inputs = self.iactivation_function(hidden_outputs)
        # 计算输入层的数据
        inputs = np.dot(self.wih.T, hidden_inputs)
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 255
        print(inputs)
        return inputs


    # 神经网络测试
    def test(self, inputs_list):
        # 将输入数据转化成二维矩阵
        inputs = np.array(inputs_list, ndmin=2).T

        # 计算隐层的输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐层的输出
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的输出
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# 训练神经网络
def train_neuralnetwork():
    # 初始化 784（28 * 28）个输入节点，100个隐层节点，10个输出节点（0~9）
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # 学习率0.3
    learning_rate = 0.1
    # 训练次数
    epochs = 5
    # 初始化神经网络实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读取训练集
    training_data_file = open("D:/python study/shuzi/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # 训练数据
    for e in range(epochs):
        for record in training_data_list[1:]:
            all_values = record.split(',')
            # 输入数据范围（0.01~1）
            inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
            # 标记数据（相应标记为0.99，其余0.01）
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    np.savetxt('D:/python study/shuzi/who.csv', n.who, delimiter=',')
    np.savetxt('D:/python study/shuzi/wih.csv', n.wih, delimiter=',')

# 验证性能
def pre_acc():

    # 初始化 784（28 * 28）个输入节点，100个隐层节点，10个输出节点（0~9）
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # 学习率0.1
    learning_rate = 0.1

    # 初始化神经网络实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读取测试数据
    test_data_file = open("D:/python study/shuzi/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # 性能分析（正确率）
    wih = np.loadtxt(open('D:/python study/shuzi/wih.csv'), delimiter=",", skiprows=0)
    who = np.loadtxt(open('D:/python study/shuzi/who.csv'), delimiter=",", skiprows=0)
    n.setweights(wih, who)
    scorecard = []
    for record in test_data_list[1:]:
        # 读取测试集中的每个数据，进行预测
        test_data = record.split(',')
        correct_label = int(test_data[0])
        outputs = n.test(np.asfarray(test_data[1:]) / 255.0 * 0.99 + 0.01)
        # 找到每次预测结果中最大值的索引（即，预测的标签）
        pre_label = np.argmax(outputs)
        # 如果预测正确，则得1分，否则得0分
        if pre_label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    print(scorecard)
    # 计算正确率
    scorecard_array = np.asarray(scorecard)
    accuracy = scorecard_array.sum() / scorecard_array.size
    print(accuracy)

# 单个预测
def pre_one():
    # 初始化 784（28 * 28）个输入节点，100个隐层节点，10个输出节点（0~9）
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # 学习率0.1
    learning_rate = 0.1
    # 初始化神经网络实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读取测试数据
    test_data_file = open('D:/python study/shuzi//mnist_test.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # 打印测试数据标签
    test_data = test_data_list[1].split(',')
    print('原标签：', test_data[0])

    # 生成标签图片
    image_array = np.asfarray(test_data[1:]).reshape(28, 28)
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()

    # 利用神经网络预测
    wih = np.loadtxt(open('D:/python study/shuzi/wih.csv'), delimiter=",", skiprows=0)
    who = np.loadtxt(open('D:/python study/shuzi/who.csv'), delimiter=",", skiprows=0)
    n.setweights(wih, who)
    results = n.test(np.asfarray(test_data[1:]) / 255.0 * 0.99 + 0.01)
    pre_label = np.argmax(results)
    print('预测结果：', pre_label)
    print(results)

# 根据标签逆向生成图片
def pre_imakepic():
    # 初始化 784（28 * 28）个输入节点，100个隐层节点，10个输出节点（0~9）
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # 学习率0.1
    learning_rate = 0.1
    # 初始化神经网络实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 要制作的标签
    want_label = 0
    # 生成相应的输出层的输出数据
    label_data = np.zeros(output_nodes) + 0.01
    label_data[int(want_label)] = 0.99
    wih = np.loadtxt(open('D:/python study/shuzi/wih.csv'), delimiter=",", skiprows=0)
    who = np.loadtxt(open('D:/python study/shuzi/who.csv'), delimiter=",", skiprows=0)
    n.setweights(wih, who)
    res_inputs = n.makeinput(label_data)

    # 生成标签图片
    image_array = np.asfarray(res_inputs).reshape(28, 28)
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()

# 手写照片识别
def pre_pic():
    # 初始化 784（28 * 28）个输入节点，100个隐层节点，10个输出节点（0~9）
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # 学习率0.1
    learning_rate = 0.1
    # 初始化神经网络实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读取带预测图片
    img = Image.open('D:/python study/shuzi/2.jpg').convert('L')
    newimg = img.resize((28, 28), Image.ADAPTIVE)
    # newimg.save('picture/2_resize.jpg')
    test_pic = np.array(newimg)

    # 利用神经网络预测
    wih = np.loadtxt(open('D:/python study/shuzi/wih.csv'), delimiter=",", skiprows=0)
    who = np.loadtxt(open('D:/python study/shuzi/who.csv'), delimiter=",", skiprows=0)
    n.setweights(wih, who)
    # 正常255表示白色，但mnist数据集255表示黑色，所以现实图片颜色应该翻转一下
    results = n.test(np.asfarray(255.0 - test_pic.flatten()) / 255.0 * 0.99 + 0.01)
    pre_label = np.argmax(results)
    print('预测结果：', pre_label)
    print(results)

"""
if __name__ == "__main__":
    # train_neuralnetwork()
    # pre_pic()
    # pre_imakepic()
    # pre_one()
    # pre_acc()
"""