import os
from numpy import matlib
import mne
from CLO import CLO
from Global_Vars import Global_Vars
from MFO import MFO
from Model_CNN import Model_GoogleNet_CNN, Model_CNN_Pred
from Model_DCNN import Model_DCNN, Model_IDCNNPred
from Model_GAN_GRU import Model_NDAGAT_GRU
from Model_RNN import Model_RNN
from Model_SVM import Model_SVM
from Model_TCN import Model_TCN
from NGO import NGO
from Objfun import objfun, objfun_1
from Plot_Results import *
from Proposed import Proposed
from QSO import QSO
from Sample_Results import Sample_Signal


def Read_Text(Dir_filename, FileName):
    fid = open(Dir_filename, 'r', encoding='cp1252')  # Open Text file
    Text = fid.readlines()  # Read Text File
    for i in range(len(Text)):
        if Text[i][11:-1] == FileName.split('/')[5][:-9]:
            if Text[i + 3][0:6] == 'Number':  # Get the Classes Stored Variable
                Label = int(Text[i + 3][len(Text[i + 3]) - 2])  # Get the Classes
                return Label


def Read_Dataset(Directory):
    Data = []
    Seizures = []
    Tar = []
    listFolds = os.listdir(Directory)  # List of Directory
    for i in range(6, len(listFolds)):
        folderName = Directory + listFolds[i]
        if os.path.isdir(folderName):
            listFiles = os.listdir(folderName)
            for j in range(len(listFiles)):
                try:
                    if '.txt' not in listFiles[j] and '.seizures' not in listFiles[j] and \
                            listFiles[
                                j] + '.seizures' in listFiles:  # Only .seizures and .txt files should be considered. Check the condition first, then retrieve the file we need
                        filename = folderName + '/' + listFiles[j]  # Get FileName with root
                        rawData = mne.io.read_raw_edf(filename)  # Read file
                        rawData = rawData.get_data()  # Get Data
                        filename = folderName + '/' + listFiles[j] + '.2000'
                        File_Split = listFiles[0].split('.')
                        if File_Split[len(File_Split) - 1] == 'txt':
                            Label = Read_Text(folderName + '/' + listFiles[0], filename)  # Read and Retrieve Classes
                            Tar.append(Label)  # append Classes
                        seizuresPeroid = [38, 41, 49]
                        seizuresFileObj = open(filename, 'r')
                        Text = ''.join(seizuresFileObj.readlines())
                        firstSeiz = bin(ord(Text[seizuresPeroid[0]])).replace("0b", "")
                        secondSeiz = bin(ord(Text[seizuresPeroid[1]])).replace("0b", "")
                        seizureStartTime = int(firstSeiz + secondSeiz, 2)
                        seizureLengthInSeconds = ord(Text[seizuresPeroid[2]])
                        Data.append(rawData)
                        Seizures.append(seizureStartTime)
                except:
                    continue
    return Data, Seizures, Tar


# Read Dataset
an = 0
if an == 1:
    Data, Seizures, Tar = Read_Dataset(
        './Dataset/Dataset1/chb-mit-scalp-eeg-database-1.0.0/')  # Provide the root path in the read function, then
    # retrieve the signal and the classes
    np.save('Signal.npy', np.asarray(Data))  # Save Signal
    np.save('Target.npy', np.reshape(Tar, (-1, 1)))  # Save Target
    np.save('Seizures_loc.npy', np.reshape(np.asarray(Seizures), (-1, 1)))  # Save Seizures_loc

# Optimization for Classification
an = 0
if an == 1:
    Data = np.load('Signal.npy', allow_pickle=True)  # Load Signal
    Target = np.load('Target.npy', allow_pickle=True)  # Load Target
    Global_Vars.Data = Data  # Data Assign in Global for data use in Other file like Object Fun
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 4  # Hidden neuron count, Learning Rate (GAN and GRU)
    xmin = matlib.repmat([5, 0.01, 5, 0.01], Npop, 1)  # Set Minimum limits for Parameter turning
    xmax = matlib.repmat([255, 0.99, 255, 0.99], Npop, 1)  # Set Maximum limits for Parameter turning
    fname = objfun
    initsol = np.zeros((Npop, Chlen))  # Generate Zero array based on Npop and Chlen
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])  # Initsol value get and Store in zero array
    Max_iter = 50

    print("QSO...")
    [bestfit1, fitness1, bestsol1, time1] = QSO(initsol, fname, xmin, xmax, Max_iter)

    print("MFO...")
    [bestfit2, fitness2, bestsol2, time2] = MFO(initsol, fname, xmin, xmax, Max_iter)

    print("NGO...")
    [bestfit3, fitness3, bestsol3, time3] = NGO(initsol, fname, xmin, xmax, Max_iter)

    print("CLO...")
    [bestfit4, fitness4, bestsol4, time4] = CLO(initsol, fname, xmin, xmax, Max_iter)

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    Fitness = [fitness1, fitness2, fitness3, fitness4, fitness5]
    np.save('BestSol.npy', BestSol)  # Save BestSol
    np.save('Fitness.npy', Fitness)  # Save Fitness

# Classification
an = 0
if an == 1:
    Feat = np.load('Signal.npy', allow_pickle=True)  # loading step
    Target = np.load('Target.npy', allow_pickle=True)  # loading step
    BestSol = np.load('BestSol.npy', allow_pickle=True)  # loading step
    K = 5  # 5 Means Kfold for Comparison
    Per = 1 / 5
    Perc = round(Feat.shape[0] * Per)  # Splitting Data
    Fold = []
    for i in range(K):
        Eval = np.zeros((10, 25))  # Generate Zero array for 10 is Algorithm + Method and 25 is Metrics
        for j in range(BestSol.shape[0]):
            sol = np.round(BestSol[j, :]).astype(np.int16)  # Get Sol value in Integer
            # Split Training, Testing Datas
            Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Feat.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Feat[train_index, :]
            Train_Target = Target[train_index, :]
            Eval[j, :], pred, map = Model_NDAGAT_GRU(Train_Data, Train_Target, Test_Data, Test_Target,
                                                     sol)  # With Optimization
        Eval[5, :], pred1 = Model_GoogleNet_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :], pred2 = Model_DCNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :], pred3 = Model_SVM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :], pred4 = Model_NDAGAT_GRU(Train_Data, Train_Target, Test_Data,
                                             Test_Target)  # Without Optimization
        Eval[9, :] = Eval[4, :]
        Fold.append(Eval)
    np.save('Evaluate_all.npy', Fold)  # Save Eval all

# Optimization for Prediction
an = 0
if an == 1:
    Data = np.load('Signal.npy', allow_pickle=True)  # loading step
    Target = np.load('Seizures_loc.npy', allow_pickle=True)  # loading step
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden neuron count, Learning Rate, No. of Epochs
    xmin = matlib.repmat([5, 0.01, 5], Npop, 1)  # Set Minimum limits for Parameter turning
    xmax = matlib.repmat([255, 0.99, 50], Npop, 1)  # Set Maximum limits for Parameter turning
    fname = objfun_1
    initsol = np.zeros((Npop, Chlen))  # Generate Zero array based on Npop and Chlen
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])  # Initsol value get and Store in zero array
    Max_iter = 50

    print("QSO...")
    [bestfit1, fitness1, bestsol1, time1] = QSO(initsol, fname, xmin, xmax, Max_iter)

    print("MFO...")
    [bestfit2, fitness2, bestsol2, time2] = MFO(initsol, fname, xmin, xmax, Max_iter)

    print("NGO...")
    [bestfit3, fitness3, bestsol3, time3] = NGO(initsol, fname, xmin, xmax, Max_iter)

    print("CLO...")
    [bestfit4, fitness4, bestsol4, time4] = CLO(initsol, fname, xmin, xmax, Max_iter)

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    Fitness = [fitness1, fitness2, fitness3, fitness4, fitness5]
    np.save('Sol.npy', BestSol)  # Save Files
    np.save('Fit.npy', Fitness)

# Prediction
an = 0
if an == 1:
    Sig = np.load('Signal.npy', allow_pickle=True)  # loading step
    Target = np.load('Seizures_loc.npy', allow_pickle=True)  # loading step
    BestSol = np.load('Sol.npy', allow_pickle=True)  # loading step
    EVAL = []
    Step_per_Epoch = [100, 200, 300, 400]
    for act in range(len(Step_per_Epoch)):
        learnperc = round(Sig.shape[0] * 0.75)  # Split Training and Testing Datas
        Train_Data = Sig[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Sig[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((10, 7))
        for j in range(BestSol.shape[0]):
            print(act, j)
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], pred = Model_TCN(Train_Data, Train_Target, Test_Data, Test_Target,
                                         Step_per_Epoch[act], sol)  # With optimization
        Eval[5, :], pred1 = Model_CNN_Pred(Train_Data, Train_Target, Test_Data, Test_Target,
                                           Step_per_Epoch[act])
        Eval[6, :], pred2 = Model_RNN(Train_Data, Train_Target, Test_Data,
                                      Test_Target, Step_per_Epoch[act])
        Eval[7, :], pred3 = Model_IDCNNPred(Train_Data, Train_Target, Test_Data, Test_Target,
                                            Step_per_Epoch[act])
        Eval[8, :], pred4 = Model_TCN(Train_Data, Train_Target, Test_Data, Test_Target,
                                      Step_per_Epoch[act])  # Without optimization
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_all.npy', EVAL)  # Save Eval all

# Plot Files
plotConvResults()
Plot_ROC_Curve()
Plots_Results()
Line_PlotTesults()
Proposed_PlotResults()
Table()
Pred_ConvResults()
Pred_Plot_Results()
Pred_Table()
Sample_Signal()
