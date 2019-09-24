from weights.neural_network import neuralNetwork, train_neuralnetwork
import numpy as np
import pandas as pd

def pre_kgtest():
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # 学习率0.1
    learning_rate = 0.1
    # 初始化神经网络实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读取测试数据
    test = pd.read_csv("D:/python study/shuzi/test.csv")
    test_data_file = open('D:/python study/shuzi//test.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    wih = np.loadtxt(open('D:/python study/shuzi/wih.csv'), delimiter=",", skiprows=0)
    who = np.loadtxt(open('D:/python study/shuzi/who.csv'), delimiter=",", skiprows=0)
    n.setweights(wih, who)
    pred = []

    # 预测数据
    for record in test_data_list[1:]:
        test_data = record.split(",")
        inputs = np.asfarray(test_data) / 255.0 * 0.99 + 0.01
        results = n.test(inputs)
        pre_label = np.argmax(results)
        pred.append(pre_label)
    return pred

if __name__ == "__main__":

    prediction = pre_kgtest()
    prediction = np.asarray(prediction)
    result = pd.DataFrame({"ImageId": range(1, 28001), "Label": prediction})
    result.to_csv("D:/python study/shuzi/kaggle_pre.csv")
    print(result.head(10))
    print(result.info())