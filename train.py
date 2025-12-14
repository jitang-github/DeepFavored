"""
* DeepFavored - A deep learning based method to identify favored(adaptive) mutations in population genomes.
* Copyright (C) 2022 Ji Tang, Hao Zhu
*
* This program is licensed for academic research use only
* unless otherwise stated. Contact jitang1024@outlook.com, zhuhao@smu.edu.cn for
* commercial licensing options.
*
* For use in any publications please cite: Ji Tang, Maosheng Huang, Sha He, Junxiang Zeng, Hao Zhu (2022). Uncovering the extensive trade-off between adaptive evolution and disease susceptibility. Cell Reports, Volume 40, Issue 11, 111351.
"""

import json
import os
import random
import re
import warnings
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from network import MLPtwoOut
from utils import shuffle, WriteToLogFile, format_time_differ, PlotPowerFDRcurve, \
    RecurReadFilePaths, PlotRankCDFcurve, MhtPlot
from identify import Identify


# ====================================================================================================
# tools for training
# {
def WrapLossFunc(lossFuncName, Y, Y_pred):
    loss = ''
    if lossFuncName == 'BinaryCrossentropy':
        loss = tf.keras.losses.binary_crossentropy(Y, Y_pred)
    return loss


def WrapOpt(optName, lr_tensor):
    opt = ''
    if optName in ['RMSProp', 'Adam', 'Adadelta', 'AdagradDA', 'Adagrad', 'Ftrl', 'GradientDescent', 'Momentum',
                   'ProximalAdagrad', 'ProximalGradientDescent', 'SyncReplicas']:
        opt = eval('tf.compat.v1.train.' + optName + 'Optimizer')(learning_rate=lr_tensor)
    if optName in ['AdaMax', 'Nadam']:
        opt = eval('tf.contrib.opt.' + optName + 'Optimizer')(learning_rate=lr_tensor)
    return opt


def train_valid_split(Xdata, Ydata, valid_prop):
    train_valid = []
    sampleNum = Xdata.shape[0]
    cutoff = sampleNum - round(sampleNum * valid_prop)
    for d in [Xdata, Ydata]:
        train_valid.append(d[:cutoff])
        train_valid.append(d[cutoff:])
    return train_valid


# }
# ====================================================================================================


# ====================================================================================================
# functions for loading training data
# {
def RandomSample(dataArray, sampleNum):
    dataArray = shuffle(dataArray)
    sampleTotalNum = dataArray.shape[0]
    if sampleTotalNum < sampleNum:
        info = 'There are %i muts, but %i muts are needed, thus tiggering randomly sampling with replacement.' % (
            sampleTotalNum, sampleNum)
        warnings.warn(info, UserWarning)
        dataArrayNew = list(dataArray)
        diff = sampleNum - sampleTotalNum
        for i in range(diff):
            oneSample = random.choice(dataArray)
            dataArrayNew.append(oneSample)
        dataArray = np.array(dataArrayNew)
    else:
        dataArray = dataArray[:sampleNum]
    return dataArray


def ReadStatsFile(inPath, componentStats, region_type, favMutNum=None, hitchMutNum=None, ordinaMutNum=None, ldCf=None):
    X_fav, X_hitch, X_ordina = [], [], []
    files = RecurReadFilePaths(inPath, files=[])
    print('read component stats from '+region_type+' regions ...')
    for file in tqdm(files):
        df = pd.read_csv(file, sep='\t')
        if region_type == 'neutral':
            df = df[componentStats]
            X_ordina += df.values.tolist()
        if region_type == 'sweep':
            favPOS = int(file.split('/')[-1].split('_')[0])
            df_fav = df[df.POS == favPOS]
            df_hitch = df[df.LD >= ldCf]
            df_ordina = df[df.LD < ldCf]
            df_fav = df_fav[componentStats]
            df_hitch = df_hitch[componentStats]
            df_ordina = df_ordina[componentStats]
            X_fav += df_fav.values.tolist()
            X_hitch += df_hitch.values.tolist()
            X_ordina += df_ordina.values.tolist()
    X_ordina = np.array(X_ordina)
    X_ordina = RandomSample(X_ordina, ordinaMutNum)
    if region_type == 'neutral':
        return X_ordina
    if region_type == 'sweep':
        X_fav, X_hitch = np.array(X_fav), np.array(X_hitch)
        X_fav, X_hitch = RandomSample(X_fav, favMutNum), RandomSample(X_hitch, hitchMutNum)
        return X_fav, X_hitch, X_ordina


def GenerateLabel(rowsNum, positive):
    if positive:
        one_label = [1.0, 0.0]
    else:
        one_label = [0.0, 1.0]
    labels = [one_label] * rowsNum
    labels = np.array(labels)
    return labels


def LoadTrainData(trainDataPath, argsDict, shuffleSamples=False, save_npz=True):
    print('### Load training data ...')
    X_H, Y_H, X_O, Y_O = '', '', '', ''
    npzFile = trainDataPath + '/trainData.npz'
    alreadyLoad = False
    if save_npz:
        if os.path.exists(npzFile):
            npzData = np.load(npzFile, allow_pickle=True)
            X_H, Y_H, X_O, Y_O = npzData['X_H'], npzData['Y_H'], npzData['X_O'], npzData['Y_O']
            alreadyLoad = True
    if not alreadyLoad:
        # read component statistics
        stats = argsDict['componentStats']
        favMutNum, hitchMutNum, ordinaMutNum, ldCf = argsDict['favMutNum'], argsDict['hitchMutNum'], argsDict[
            'ordinaMutNum'], argsDict['LDcutoff']
        ordinaMutNum_sweep, ordinaMutNum_neut = round(ordinaMutNum / 2), round(ordinaMutNum / 2)
        X_fav, X_hitch, X_ordina = ReadStatsFile(trainDataPath + '/sweep_regions',
                                                 componentStats=stats,
                                                 region_type='sweep',
                                                 favMutNum=favMutNum,
                                                 hitchMutNum=hitchMutNum,
                                                 ordinaMutNum=ordinaMutNum_sweep,
                                                 ldCf=ldCf
                                                 )
        X_ordina_neut = ReadStatsFile(trainDataPath + '/neutral_regions', componentStats=stats, region_type='neutral',
                                      ordinaMutNum=ordinaMutNum_neut
                                      )
        X_ordina = np.vstack((X_ordina, X_ordina_neut))

        # generate label
        Y_fav = GenerateLabel(rowsNum=X_fav.shape[0], positive=True)
        Y_hitch = GenerateLabel(rowsNum=X_hitch.shape[0], positive=False)
        Y_ordina = GenerateLabel(rowsNum=X_ordina.shape[0], positive=False)

        # compose training data
        # fav muts and hitch muts
        X_H = np.vstack((X_fav, X_hitch))
        Y_H = np.vstack((Y_fav, Y_hitch))

        # fav muts and ordinary neut muts
        X_O = np.vstack((X_fav, X_ordina))
        Y_O = np.vstack((Y_fav, Y_ordina))

        if save_npz:
            np.savez(npzFile.rstrip('.npz'), X_H=X_H, Y_H=Y_H, X_O=X_O, Y_O=Y_O)

    if shuffleSamples:
        X_H, Y_H = shuffle(X_H, Y_H)
        X_O, Y_O = shuffle(X_O, Y_O)
    return X_H, Y_H, X_O, Y_O


# }
# ====================================================================================================


def Train(model, train_data, epochs, batch_size, loss, lr, reduce_lr_epochs, early_stop_epochs,
          optimizer_H, optimizer_O, modelOutDir, valid_prop=0.1, log_file=None):
    """
    Alternatively train classifiers H and O
    :param model: classifiers H and O to be trained
    :param train_data: training data for classifiers H and O
    :param epochs: max epochs
    :param batch_size: batch size for mini-batch training
    :param loss: loss function
    :param lr: learning rate
    :param reduce_lr_epochs: when validation loss do not decrease through consecutively the epochs, lr reduce
    :param early_stop_epochs:when validation loss do not decrease through consecutively the epochs, training stop
    :param optimizer_H: optimizer for classifier H
    :param optimizer_O: optimizer for classifier O
    :param modelOutDir: directory saving trained model
    :param valid_prop: proportion of validation data in train_data
    :param log_file: file logging training information
    :return:
    """
    log_info = '### Training start ...'
    print(log_info)
    if log_file:
        WriteToLogFile(log_file, log_info)

    timeStart = time.time()
    tf.compat.v1.disable_eager_execution()

    # shuffle data and split it into train and validation set
    X1_data, Y1_data, X2_data, Y2_data = train_data
    X1_data, Y1_data = shuffle(X1_data, Y1_data)
    X2_data, Y2_data = shuffle(X2_data, Y2_data)
    X1_train, X1_valid, Y1_train, Y1_valid = train_valid_split(X1_data, Y1_data, valid_prop)
    X2_train, X2_valid, Y2_train, Y2_valid = train_valid_split(X2_data, Y2_data, valid_prop)

    X1_train_sampleNum, X2_train_sampleNum = X1_train.shape[0], X2_train.shape[0]
    sampleTotalNum = X1_train_sampleNum if X1_train_sampleNum < X2_train_sampleNum else X2_train_sampleNum

    y1_classNum, y2_classNum = Y1_data.shape[1], Y2_data.shape[1]

    X, H_pred, O_pred = model.build()
    H = tf.compat.v1.placeholder("float", [None, y1_classNum], name="H_true")
    O = tf.compat.v1.placeholder("float", [None, y2_classNum], name="O_true")

    # loss
    H_loss = WrapLossFunc(loss, H, H_pred)
    O_loss = WrapLossFunc(loss, O, O_pred)

    # learning rate
    lr_tensor = tf.Variable(lr, shape=[], trainable=False)
    new_lr_tensor = tf.compat.v1.placeholder(tf.float32, shape=[], name="new_lr_tensor")
    update_lr = tf.compat.v1.assign(lr_tensor, new_lr_tensor)

    # optimizer
    optH = WrapOpt(optimizer_H, lr_tensor)
    optO = WrapOpt(optimizer_O, lr_tensor)
    trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    H_opt = optH.minimize(H_loss, var_list=trainable_vars)
    O_opt = optO.minimize(O_loss, var_list=trainable_vars)

    trainedModelDir = modelOutDir + '/model'
    os.makedirs(trainedModelDir, exist_ok=True)
    best_model_file = trainedModelDir + '/model'

    # run training and log training info
    saver = tf.compat.v1.train.Saver(defer_build=False)
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        best_loss_epoch_tuple = (-1, 0)  # initialization
        count_for_training_on_best = 0
        new_lr = lr
        for epoch in range(1, epochs + 1):
            log_info = '==='
            print(log_info)
            if log_file:
                WriteToLogFile(log_file, log_info)
            batch_start, batch_end = 0, batch_size
            step = 0
            info_step = int((sampleTotalNum / 3) / batch_size)
            while batch_end < sampleTotalNum:
                X1_batch, Y1_batch = X1_train[batch_start:batch_end], Y1_train[batch_start:batch_end]
                X2_batch, Y2_batch = X2_train[batch_start:batch_end], Y2_train[batch_start:batch_end]
                hopt, H_batch_loss = session.run([H_opt, H_loss], feed_dict={X: X1_batch, H: Y1_batch})
                oopt, O_batch_loss = session.run([O_opt, O_loss], feed_dict={X: X2_batch, O: Y2_batch})

                batch_start = batch_end
                batch_end += batch_size
                step += 1
                if step % info_step == 0:
                    log_info = 'Epoch %d/%d %d/%d, loss: H-%f O-%f' % (
                        epoch, epochs, step * batch_size, sampleTotalNum, H_batch_loss.mean(), O_batch_loss.mean())
                    print(log_info)
                    if log_file:
                        WriteToLogFile(log_file, log_info)

            H_train_loss = session.run(H_loss, feed_dict={X: X1_train, H: Y1_train})
            O_train_loss = session.run(O_loss, feed_dict={X: X2_train, O: Y2_train})
            H_valid_loss = session.run(H_loss, feed_dict={X: X1_valid, H: Y1_valid})
            O_valid_loss = session.run(O_loss, feed_dict={X: X2_valid, O: Y2_valid})
            H_train_loss, O_train_loss = H_train_loss.mean(), O_train_loss.mean()
            H_valid_loss, O_valid_loss = H_valid_loss.mean(), O_valid_loss.mean()
            log_info = 'Epoch %d/%d, train_loss: H-%f  O-%f  valid_loss: H-%f  O-%f' % (
                epoch, epochs, H_train_loss, O_train_loss, H_valid_loss, O_valid_loss)
            print(log_info)
            if log_file:
                WriteToLogFile(log_file, log_info)

            valid_loss = H_valid_loss + O_valid_loss  # can have different weights
            best_valid_loss, best_epoch = best_loss_epoch_tuple
            if best_valid_loss is -1 or valid_loss < best_valid_loss:
                log_info = 'Decrease valid_loss from %f to %f, save the newest model.' % (best_valid_loss, valid_loss)
                print(log_info)
                if log_file:
                    WriteToLogFile(log_file, log_info)
                valid_loss_e04 = round(valid_loss, 4)
                best_loss_epoch_tuple = (valid_loss, epoch)
                best_H_valid_loss, best_O_valid_loss = H_valid_loss, O_valid_loss
                saver.save(session, best_model_file)
            else:
                log_info = 'Not decrease valid_loss.'
                print(log_info)
                if log_file:
                    WriteToLogFile(log_file, log_info)
                count_for_training_on_best += 1
                if count_for_training_on_best > 3:
                    # continue to train based on the values of trainable variables of the saved best model
                    saver.restore(session,
                                  best_model_file)  # all of the values of saved (trained) variables have been restored
                    log_info = 'Training based on the values of trainable variables of the saved best model.'
                    print(log_info)
                    if log_file:
                        WriteToLogFile(log_file, log_info)
                    count_for_training_on_best = 0

                if epoch - best_epoch > reduce_lr_epochs:
                    # Update learning rate
                    old_lr = new_lr
                    new_lr = old_lr * 0.5
                    session.run(update_lr, feed_dict={new_lr_tensor: new_lr})
                    # opt.lr = new_lr
                    log_info = 'Reduce lr from %f to %f.' % (old_lr, new_lr)
                    print(log_info)
                    if log_file:
                        WriteToLogFile(log_file, log_info)

                if epoch - best_epoch > early_stop_epochs:
                    # Stop training
                    log_info = 'EarlyStopping!'
                    print(log_info)
                    if log_file:
                        WriteToLogFile(log_file, log_info)
                    break
        timeEnd = time.time()
        log_info = '### Training done! Taking %s. Epoch %d has the best valid_loss %f(H_valid_loss:%f, O_valid_loss:%f).' \
                   % (format_time_differ(timeEnd-timeStart), best_loss_epoch_tuple[1], best_loss_epoch_tuple[0],
                      best_H_valid_loss, best_O_valid_loss)
        print(log_info)
        if log_file:
            WriteToLogFile(log_file, log_info)


def TrainWithArgs(args):
    with open(args.hyparams) as f:
        hyparams = json.load(f)

    # save all of the hyper-params for the model to be trained
    os.makedirs(args.modelDir, exist_ok=True)
    hyparams['trainDataHyparams']['trainData'] = os.path.abspath(args.trainData)
    with open(args.modelDir + '/hyperparams.json', 'w') as f:
        f.write(json.dumps(hyparams, sort_keys=True, indent=4))

    model_hyparams, train_hyparams, train_data_hyparams = hyparams['modelHyparams'], hyparams['trainHyparams'], \
                                                          hyparams['trainDataHyparams']

    # load training data
    X1_train, Y1_train, X2_train, Y2_train = LoadTrainData(args.trainData, train_data_hyparams)

    # load model
    featureNum = X1_train.shape[1]
    model = MLPtwoOut(featuresNum=featureNum,
                      batchNormHiddenLayer=train_hyparams['hidden_batchnorm'],
                      init=train_hyparams['init'],
                      hiddenLayerActFunc=model_hyparams['hiddenLayerActFunc'],
                      sharedLayers=model_hyparams['sharedLayers'],
                      Y1_hiddenLayers=model_hyparams['H_hiddenLayers'],
                      Y2_hiddenLayers=model_hyparams['O_hiddenLayers'],
                      Y1_outputLayerNodeNum=model_hyparams['H_outputLayerNodeNum'],
                      Y1_outputLayerActFunc=model_hyparams['H_outputLayerActFunc'],
                      Y2_outputLayerNodeNum=model_hyparams['O_outputLayerNodeNum'],
                      Y2_outputLayerActFunc=model_hyparams['O_outputLayerActFunc']
                      )
    # train model
    training_log_file = args.modelDir + '/training.log'
    train_cmd = 'python DeepFavored.py train --hyparams %s --modelDir %s --trainData %s ' % (args.hyparams,
                                                                                             args.modelDir,
                                                                                             args.trainData)
    if 'testData' in args:
        train_cmd += ' --testData '+args.testData
    WriteToLogFile(training_log_file, train_cmd)
    Train(model=model,
          train_data=(X1_train, Y1_train, X2_train, Y2_train),
          epochs=train_hyparams['epochs'],
          batch_size=train_hyparams['batch_size'],
          loss=train_hyparams['loss'],
          lr=train_hyparams['lr'],
          reduce_lr_epochs=train_hyparams['reduce_lr_epochs'],
          early_stop_epochs=train_hyparams['early_stop_epochs'],
          optimizer_H=train_hyparams['optimizer_H'],
          optimizer_O=train_hyparams['optimizer_O'],
          modelOutDir=args.modelDir,
          log_file=training_log_file
          )

    # performance evaluation
    if 'testData' in args:
        print('### Performance evaluation start ...')
        performanceDir = args.modelDir + '/performance'
        os.makedirs(performanceDir, exist_ok=True)
        if 'rankCDF_powerFDR_data' in os.listdir(args.testData):
            dfOutPath = args.testData+'/rankCDF_powerFDR_data.df.out'
            os.makedirs(dfOutPath, exist_ok=True)
            files = RecurReadFilePaths(args.testData + '/rankCDF_powerFDR_data', files=[])
            df_in_out_files = []
            for infile in files:
                outfile = re.sub('/rankCDF_powerFDR_data/', '/rankCDF_powerFDR_data.df.out/', infile)
                outPath = '/'.join(outfile.split('/')[:-1])
                os.makedirs(outPath, exist_ok=True)
                df_in_out_files.append((infile, outfile))
            Identify(args.modelDir, df_in_out_files)
            PlotRankCDFcurve(dfOutPath, performanceDir+'/rankCDFcurve.png')
            PlotPowerFDRcurve(dfOutPath, performanceDir + '/powerFDRcurve.png')
        if 'mht_plot_data' in os.listdir(args.testData):
            dfOutPath = args.testData+'/mht_plot_data.df.out'
            os.makedirs(dfOutPath, exist_ok=True)
            files = RecurReadFilePaths(args.testData + '/mht_plot_data', files=[])
            df_in_out_files = []
            for infile in files:
                outfile = re.sub('/mht_plot_data/', '/mht_plot_data.df.out/', infile)
                outPath = '/'.join(outfile.split('/')[:-1])
                os.makedirs(outPath, exist_ok=True)
                df_in_out_files.append((infile, outfile))
            Identify(args.modelDir, df_in_out_files)
            MhtPlot(dfOutPath, performanceDir + '/mht_plot')
        print('### Performance evaluation done!')


if __name__ == '__main__':
    pass
