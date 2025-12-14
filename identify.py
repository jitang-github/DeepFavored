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

import pandas as pd
import tensorflow as tf
from tqdm import tqdm


def Identify(modelDir, inOutFiles):
    print('### Run DeepFavored ...')
    with open(modelDir + '/hyperparams.json') as f:
        confArgs = json.load(f)
    features = confArgs['trainDataHyparams']['componentStats']
    with tf.Graph().as_default() as g:  # use tf.Graph().as_default() to separate differnt Scope
        with tf.compat.v1.Session() as sess:
            saver = tf.compat.v1.train.import_meta_graph(modelDir + '/model/model.meta')  # , clear_devices=True)
            saver.restore(sess, modelDir + '/model/model')
            graph = tf.compat.v1.get_default_graph()

            X_tensor = graph.get_tensor_by_name("X:0")
            H_tensor = graph.get_tensor_by_name("Y1_pred:0")
            O_tensor = graph.get_tensor_by_name("Y2_pred:0")

            for inFile, outFile in tqdm(inOutFiles):
                df = pd.read_csv(inFile, sep='\t')
                df.dropna(inplace=True)
                df_features = df[features]
                X = df_features.values
                positions = df.POS.values
                H, O = sess.run([H_tensor, O_tensor], feed_dict={X_tensor: X})
                rows = []
                for pos, h, o in list(zip(positions, H, O)):
                    h, o = h[0], o[0]
                    DFscore = h * o
                    coor, h, o, DFscore = int(pos), round(h, 3), round(o, 3), round(DFscore, 3)
                    rows.append((pos, h, o, DFscore))

                rows = sorted(rows, key=lambda x: x[0])
                if outFile[-7:] != '.df.out':
                    outFile += '.df.out'
                with open(outFile, 'w') as f:
                    f.write('POS\tDF_H\tDF_O\tDF\n')
                    for row in rows:
                        row = [str(i) for i in row]
                        f.write('\t'.join(row) + '\n')


def IdentifyWithArgs(args):
    inOutFiles = []
    outDir = args.outDir
    if outDir is None:
        if os.path.isfile(args.input):
            outDir = args.input.split('/')[0:-1]
        else:
            outDir = args.input
    os.makedirs(outDir, exist_ok=True)
    if os.path.isfile(args.input):
        fileName = args.input.split('/')[-1]
        inOutFiles.append((args.input, outDir+'/'+fileName))
    else:
        for fileName in os.listdir(args.input):
            inFile = args.input + '/' + fileName
            outFile = outDir + '/' + fileName
            inOutFiles.append((inFile, outFile))
    Identify(args.modelDir, inOutFiles)
