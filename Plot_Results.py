import numpy as np
import warnings
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pylab

warnings.filterwarnings("ignore")


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'QSO-NDAGAT-GRU', 'MFO-NDAGAT-GRU', 'NGO-NDAGAT-GRU', 'CLO-NDAGAT-GRU', 'FNG-CLO-NDAGAT-GRU']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    length = np.arange(Fitness.shape[1])
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve for Classification')
    plt.plot(length, Fitness[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label=Algorithm[1])
    plt.plot(length, Fitness[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label=Algorithm[2])
    plt.plot(length, Fitness[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label=Algorithm[3])
    plt.plot(length, Fitness[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label=Algorithm[4])
    plt.plot(length, Fitness[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label=Algorithm[5])
    plt.xlabel('No. of Iteration', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylabel('Cost Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.legend(loc=1, prop={'weight': 'bold', 'size': 12})
    plt.savefig("./Results/Conv.png")
    plt.show()


def Pred_ConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fit.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'QSO-RA-ATCN', 'MFO-RA-ATCN', 'NGO-RA-ATCN', 'CLO-RA-ATCN', 'FNG-CLO-RA-ATCN']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('--------------------------------------------------Prediction Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    length = np.arange(Fitness.shape[1])
    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve for Prediction')
    plt.plot(length, Fitness[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label=Algorithm[1])
    plt.plot(length, Fitness[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label=Algorithm[2])
    plt.plot(length, Fitness[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label=Algorithm[3])
    plt.plot(length, Fitness[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label=Algorithm[4])
    plt.plot(length, Fitness[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label=Algorithm[5])
    plt.xlabel('No. of Iteration', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylabel('Cost Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.legend(loc=1, prop={'weight': 'bold', 'size': 12})
    plt.savefig("./Results/Pred_Conv.png")
    plt.show()


def Plot_ROC_Curve():
    lw = 3
    cls = ['Google-net CNN', 'DCNN', 'SVM', 'GAT-GRU', 'FNG-CLO-NDAGAT-GRU']
    Actual = np.load('Target.npy', allow_pickle=True)
    lenper = round(Actual.shape[0] * 0.25)
    Actual = Actual[lenper:, :]
    colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
    fig = plt.figure()
    fig.canvas.manager.set_window_title('ROC Curve')
    for i, color in zip(range(5), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc * 100

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            lw=2,
            label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylabel("True Positive Rate", fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='#1d3557')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#1d3557')
    plt.title("ROC Curve")
    plt.legend(loc="lower right", prop={'weight': 'bold', 'size': 12})
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Table():
    eval = np.load('Evaluate_Act.npy', allow_pickle=True)
    Algorithm = ['Activation Function', 'QSO-NDAGAT-GRU', 'MFO-NDAGAT-GRU', 'NGO-NDAGAT-GRU', 'CLO-NDAGAT-GRU',
                 'FNG-CLO-NDAGAT-GRU']
    Classifier = ['Activation Function', 'Google-net CNN', 'DCNN', 'SVM', 'GAT-GRU', 'FNG-CLO-NDAGAT-GRU']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 2, 4, 6, 9, 15]).astype(int)
    Table_Terms = [0, 2, 4, 6, 9, 15]
    table_terms = [Terms[i] for i in Table_Terms]
    Activation_Fun = ['Linear', 'Relu', 'Tanh', 'Softmax', 'Sigmoid']
    for k in range(len(Table_Terms)):
        value = eval[:, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Activation_Fun)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
        print('---------------------------------- ', table_terms[k], '  Algorithm Comparison',
              '---------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Activation_Fun)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
        print('-------------------------------- ', table_terms[k], '  Classifier Comparison',
              '---------------------------------------')
        print(Table)


def Proposed_PlotResults():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC',
             'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'Lrplus', 'Lrminus', 'DOR', 'Prevalence']
    Graph_Term = [0, 5, 7, 9, 12]
    Batch_Size = ['4', '8', '16', '32', '48', '64', '128', '256']
    color = ['#4f5d75', '#ef8354', '#004e98', '#8e7dbe', '#2d3142', '#219ebc', 'm', '#f07167']
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], 1))
        for k in range(eval.shape[0]):
            Graph[k, 0] = eval[k, 4, Graph_Term[j] + 4]

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
        fig.canvas.manager.set_window_title('Batch Size Vs Proposed')
        ax.set_axisbelow(True)
        X = np.arange(Graph.shape[0])
        barWidth = 0.7

        bars = []
        for v in range(Graph.shape[0]):
            bar = ax.bar(X[v], Graph[v, 0], color=color[v], width=barWidth)
            bars.append(bar)

        plt.xticks(X, Batch_Size, fontname="Arial", fontsize=14, fontweight='bold', color='#003566')
        plt.xlabel('Batch Size', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#003566')
        for s in ['top', 'right']:
            plt.gca().spines[s].set_visible(False)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        path1 = "./Results/Prop_%s_bar.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()


def Plots_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    colors = ['#8f2d56', '#b388eb', '#219ebc', '#f77f00', '#8ac926']
    Classifier = ['Google-net CNN', 'DCNN', 'SVM', 'GAT-GRU', 'FNG-CLO-NDAGAT-GRU']
    Kfold = [1, 2, 3, 4, 5]
    for j in range(len(Graph_Terms)):
        Graph = np.zeros(eval.shape[0:2])
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Terms[j] + 4]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
        fig.canvas.manager.set_window_title(Terms[j] + ' Method Comparison of Kfold')
        X = np.arange(len(Kfold))
        plt.bar(X + 0.00, Graph[:, 5], color='#8f2d56', width=0.15, label=Classifier[0])
        plt.bar(X + 0.15, Graph[:, 6], color='#b388eb', width=0.15, label=Classifier[1])
        plt.bar(X + 0.30, Graph[:, 7], color='#219ebc', width=0.15, label=Classifier[2])
        plt.bar(X + 0.45, Graph[:, 8], color='#f77f00', width=0.15, label=Classifier[3])
        plt.bar(X + 0.60, Graph[:, 4], color='#8ac926', width=0.15, label=Classifier[4])
        plt.xticks(X + 0.30, ['1', '2', '3', '4', '5'], fontname="Arial", fontsize=14,
                   fontweight='bold', color='k')
        plt.xlabel('KFold', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)
        dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                       in colors]
        plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.12), fontsize=12,
                   frameon=False, ncol=len(Classifier), prop={'weight': 'bold', 'size': 12})
        plt.tight_layout()
        path = "./Results/%s_mod_bar.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()


def Line_PlotTesults():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    Algorithm = ['QSO-NDAGAT-GRU', 'MFO-NDAGAT-GRU', 'NGO-NDAGAT-GRU', 'CLO-NDAGAT-GRU', 'FNG-CLO-NDAGAT-GRU']
    Kfold = [1, 2, 3, 4, 5]
    for j in range(len(Graph_Terms)):
        Graph = np.zeros(eval.shape[0:2])
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Terms[j] + 4]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
        fig.canvas.manager.set_window_title(Terms[j] + ' Algorithm Comparison of Kfold')
        X = np.arange(len(Kfold))
        plt.bar(X + 0.00, Graph[:, 0], color='#6a994e', width=0.15, label=Algorithm[0])

        plt.bar(X + 0.15, Graph[:, 1], color='#00a8e8', width=0.15, label=Algorithm[1])

        plt.bar(X + 0.30, Graph[:, 2], color='violet', width=0.15, label=Algorithm[2])
        plt.bar(X + 0.45, Graph[:, 3], color='crimson', width=0.15, label=Algorithm[3])
        plt.bar(X + 0.60, Graph[:, 4], color='k', width=0.15, label=Algorithm[4])
        plt.xticks(X + 0.30, ['1', '2', '3', '4', '5'], fontsize=14, fontname="Arial",
                   fontweight='bold', color='k')
        plt.xlabel('KFold', fontsize=14, fontname="Arial", fontweight='bold', color='k')
        plt.ylabel(Terms[Graph_Terms[j]], fontsize=14, fontname="Arial", fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)
        colors = ['#6a994e', '#00a8e8', 'violet', 'crimson', 'k']
        dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                       in colors]
        plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.12), fontsize=12,
                   frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
        plt.tight_layout()
        path = "./Results/%s_Alg_bar.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()


def Pred_Plot_Results():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'Accuracy']
    Algorithm = ['QSO-RA-ATCN', 'MFO-RA-ATCN', 'NGO-RA-ATCN', 'CLO-RA-ATCN', 'FNG-CLO-RA-ATCN']
    Classifier = ['CNN', 'RNN', 'IDCNN', 'TCN', 'FNG-CLO-RA-ATCN']
    Graph_Term = [2, 3, 4, 5, 6]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Term[j]]

        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.18, 0.8, 0.65])
        fig.canvas.manager.set_window_title(Terms[j] + ' Algorithm Comparison of Step Per Epochs')
        X = np.arange(Graph.shape[0])
        ax.barh(X + 0.00, Graph[:, 0], height=0.15, color='#20B2AA', label=Algorithm[0])
        ax.barh(X + 0.15, Graph[:, 1], height=0.15, color='#E69F00', label=Algorithm[1])
        ax.barh(X + 0.30, Graph[:, 2], height=0.15, color='#80B918', label=Algorithm[2])
        ax.barh(X + 0.45, Graph[:, 3], height=0.15, color='#A52A2A', label=Algorithm[3])
        ax.barh(X + 0.60, Graph[:, 4], height=0.15, color='k', label=Algorithm[4])
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.yticks(X + 0.20, ['100', '200', '300', '400'], fontname="Arial",
                   fontsize=14,
                   fontweight='bold', color='#14213d')
        plt.ylabel('Step Per Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xlabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=12) for color
                       in ['#20B2AA', '#E69F00', '#80B918', '#A52A2A', 'k']]
        plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.468, 1.25), fontsize=10,
                   frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
        path = "./Results/%s_Alg_bar.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Step Per Epochs vs ' + Terms[Graph_Term[j]])
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.18, 0.8, 0.65])
        fig.canvas.manager.set_window_title(Terms[j] + ' Method Comparison of Step Per Epochs')
        X = np.arange(Graph.shape[0])
        ax.barh(X + 0.00, Graph[:, 5], height=0.15, color='darkmagenta', label=Classifier[0])
        ax.barh(X + 0.15, Graph[:, 6], height=0.15, color='darkorange', label=Classifier[1])
        ax.barh(X + 0.30, Graph[:, 7], height=0.15, color='darkslateblue', label=Classifier[2])
        ax.barh(X + 0.45, Graph[:, 8], height=0.15, color='#80B918', label=Classifier[3])
        ax.barh(X + 0.60, Graph[:, 4], height=0.15, color='k', label=Classifier[4])
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.yticks(X + 0.20, ['100', '200', '300', '400'], fontname="Arial",
                   fontsize=14,
                   fontweight='bold', color='#14213d')
        plt.ylabel('Step Per Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xlabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='#14213d')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        path = "./Results/%s_Mod_bar.png" % (Terms[Graph_Term[j]])
        dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=12) for color
                       in ['darkmagenta', 'darkorange', 'darkslateblue', '#80B918', 'k']]
        plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.22), fontsize=10,
                   frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
        plt.savefig(path)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Step Per Epochs vs ' + Terms[Graph_Term[j]])
        plt.show()


def Pred_Table():
    eval = np.load('Evaluate_Opt.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'Accuracy']
    Algorithm = ['Optimizer', 'QSO-RA-ATCN', 'MFO-RA-ATCN', 'NGO-RA-ATCN', 'CLO-RA-ATCN', 'FNG-CLO-RA-ATCN']
    Classifier = ['Optimizer', 'CNN', 'RNN', 'IDCNN', 'TCN', 'FNG-CLO-RA-ATCN']
    Graph_Term = np.array([0, 1, 2, 3, 4]).astype(int)
    Table_Terms = [0, 1, 2, 3, 4]
    table_terms = [Terms[i] for i in Table_Terms]
    Optimizer = ['Adam', 'SGD', 'RMSprop', 'Adamax', 'AdaGrad']
    for k in range(len(Table_Terms)):
        value = eval[:, :, :]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Optimizer)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[:, j, Graph_Term[k]])
        print('------------------------------- ', table_terms[k], '-  Predication - Algorithm Comparison',
              '---------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Optimizer)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Term[k]])
        print('------------------------------- ', table_terms[k], '- Predication - Classifier Comparison',
              '---------------------------------------')
        print(Table)


if __name__ == '__main__':
    plotConvResults()
    Plot_ROC_Curve()
    Plots_Results()
    Line_PlotTesults()
    Proposed_PlotResults()
    Table()
    Pred_ConvResults()
    Pred_Plot_Results()
    Pred_Table()
