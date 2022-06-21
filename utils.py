import os
import time
import re
import warnings
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ====================================================================================================
# generic tools
# {
def shuffle(X, Y=None):
    np.random.seed(13)
    indices = np.random.permutation(len(X))
    X = X[indices]
    if Y is None:
        return X
    else:
        Y = Y[indices]
        return X, Y


def RecurReadFilePaths(inPath, files):
    """
    Read file paths recursively
    :param inPath:
    :param files:
    :return:
    """
    if os.path.isfile(inPath):
        files.append(inPath)
    if os.path.isdir(inPath):
        for term in os.listdir(inPath):
            files = RecurReadFilePaths(inPath + '/' + term, files)
    return files


def format_time_differ(time_differ):
    hour = int(time_differ // 3600)
    minute = int((time_differ % 3600) // 60)
    second = int(time_differ % 60)
    fmt = '{:0>2d}h:{:0>2d}m:{:0>2d}s'
    return fmt.format(hour, minute, second)


def WriteToLogFile(logFile, info):
    info = '(' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ')' + info
    if info[-1] != '\n':
        info += '\n'
    logFileObj = open(logFile, 'a')
    logFileObj.write(info)
    logFileObj.close()


# }
# ====================================================================================================


# ====================================================================================================
# functions for performance evaluation
# {
def get_cutoff_list(start, end, step, decimal=3):
    term = np.arange(start, end, step)
    if start > end:
        term = np.arange(end, start, step)
        term = sorted(term, reverse=True)
    cutoff_list = [round(item, decimal) for item in term]
    return cutoff_list


def CountConfusionMtxNumbers(Y_pred, Y_true, outFile, cutoffs=None, annotation=None):
    if cutoffs is None:
        cutoffs = get_cutoff_list(1.01, -0.01, 0.01)
    headerAnnotation = '#P: number of favored mutations\n'
    headerAnnotation += '#N: number of neutral mutations\n'
    headerAnnotation += '#PP: number of the mutations that actually is favored and is predicted as favored\n'
    headerAnnotation += '#PN: number of the mutations that actually is favored but is predicted as neutral\n'
    headerAnnotation += '#NP: number of the mutations that actually is neutral but is predicted as favored\n'
    headerAnnotation += '#NN: number of the mutations that actually is neutral and is predicted as neutral\n'
    header = ['#Cutoff', 'P', 'N', 'PP', 'PN', 'NP', 'NN']
    with open(outFile, 'w') as fw:
        if annotation:
            if annotation[0] != '#':
                annotation = '#' + annotation
            if annotation[-1] != '\n':
                annotation += '\n'
            fw.write(annotation)
        fw.write(headerAnnotation)
        fw.write('\t'.join(header) + '\n')
        for c in cutoffs:
            pp, pn, n_p, nn = 0, 0, 0, 0  # n_p avoid "import numpy as np"
            for y_pred, y_true in zip(Y_pred, Y_true):
                if y_pred >= c:
                    if y_true:
                        pp += 1
                    else:
                        n_p += 1
                else:
                    if y_true:
                        pn += 1
                    else:
                        nn += 1
            row = [c, pp + pn, n_p + nn, pp, pn, n_p, nn]
            fw.write('\t'.join([str(i) for i in row]) + '\n')
    print('  save confusion matrix numbers to ' + outFile)


def CalcAuc(ratePairs, baseIdx, heightIdx):
    ratePairs.sort(key=lambda x: x[heightIdx], reverse=False)
    auc = 0
    for idx, ratePair in enumerate(ratePairs):
        base_rate, height_rate = ratePair[baseIdx], ratePair[heightIdx]
        if idx > 0:
            height = height_rate - ratePairs[idx - 1][heightIdx]
            if height > 0:
                base_bottom = ratePairs[idx - 1][baseIdx]
                base_upper = base_rate
                trapezoid_area = ((base_upper + base_bottom) * height) / 2
                auc += trapezoid_area
    return auc


def UniformDistrOnYaxis(xList, yList, cutoffs=None, y_bottom=0, y_top=1, step=0.05, decimal=3):
    xListNew, yListNew = [], []
    cutoffsNew = []

    # Generate a dict
    vals = np.arange(y_bottom, y_top + step, step)
    valsList = [round(i, decimal) for i in vals]
    valNum = len(valsList)
    noValDict = {}
    for i in range(1, valNum):
        noValDict[(valsList[i - 1], valsList[i])] = True

    for start, end in noValDict:
        for idx, y in enumerate(yList):
            if noValDict[(start, end)]:
                if start <= y <= end:
                    xListNew.append(xList[idx])
                    yListNew.append(y)
                    if cutoffs is not None:
                        cutoffsNew.append(cutoffs[idx])
                    noValDict[(start, end)] = False
            else:
                break
    if (0 not in xListNew) and (0 not in yListNew):
        xListNew, yListNew = [0] + xListNew, [0] + yListNew
        if cutoffs is not None:
            cutoffsNew = ['0'] + cutoffsNew
    if (1 not in xListNew) and (1 not in yListNew):
        xListNew, yListNew = xListNew + [1], yListNew + [1]
        if cutoffs is not None:
            cutoffsNew = cutoffsNew + ['1']
    if cutoffs is not None:
        return xListNew, yListNew, cutoffsNew
    else:
        return xListNew, yListNew


def PlotPowerFDRcurve(dfOutPath, outFig):
    print('plot power-FDR curve ...')
    confusionMtxNumsFile = '/'.join(outFig.split('/')[:-1]) + '/confusionMtxNumbers'
    Y_pred, Y_true = [], []
    dfOutFiles = RecurReadFilePaths(dfOutPath, files=[])
    for file in dfOutFiles:
        df = pd.read_csv(file, sep='\t')
        fileName = file.split('/')[-1]
        if fileName[:5] == 'sweep':
            favPos = int(fileName.split('_')[1])
            try:
                favScore = df[df.POS == favPos].DF.values[0]
                scores = df[df.POS != favPos].DF.values.tolist()
                Y_pred.append(favScore)
                Y_true.append(1)
                Y_pred += scores
                Y_true += [0] * len(scores)
            except:
                pass
        else:
            scores = df.DF.values.tolist()
            Y_pred += scores
            Y_true += [0] * len(scores)
    CountConfusionMtxNumbers(Y_pred, Y_true, confusionMtxNumsFile)
    confusionMtxNums = []
    with open(confusionMtxNumsFile, 'r') as fr:
        for row in fr.readlines():
            if row[0] != '#':
                cols = row.rstrip('\n').split('\t')
                confusionMtxNums.append([int(i) for i in cols[1:]])
    powerFDRs, tprs, fdrs = [], [], []
    all_fav_num, all_neut_num = 0, 0
    for all_fav_num, all_neut_num, pp, pn, n_p, nn in confusionMtxNums:
        predicted_P = pp + n_p
        tpr = pp / (pp + pn)  # tpr=power=recall
        if predicted_P == 0:
            fdr = 0.0
        else:
            fdr = float(n_p) / predicted_P
        powerFDRs.append((tpr, fdr))
        tprs.append(tpr)
        fdrs.append(fdr)
    powerFDRcurveAUC = CalcAuc(powerFDRs, baseIdx=0, heightIdx=1)
    fontsize = 15
    fontsize_tick_params = fontsize - 3
    line_width = 5
    fig_size = [8, 10]
    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    y_lim_bottom, y_lim_top = 0, 1

    # xList, yList = UniformDistrOnYaxis(fdrs, tprs, y_bottom=y_lim_bottom, y_top=y_lim_top)
    xList, yList = fdrs, tprs
    ax.plot(xList, yList, color='black', linestyle='-', linewidth=line_width)
    ax.set_title("AUC=%s, favored:neutral=1:%i" % (str(round(powerFDRcurveAUC, 2)), round(all_neut_num / all_fav_num)),
                 fontsize=fontsize)
    ax.locator_params(nbins=5)
    ax.set_xlabel('False Discovery Rate', fontsize=fontsize)
    ax.set_ylabel('Power', fontsize=fontsize)
    ax.set_xlim(left=0, right=1)  # , fontsize=fontsize)
    ax.set_ylim(bottom=y_lim_bottom, top=y_lim_top)  # , fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize_tick_params)
    plt.savefig(outFig, dpi=200, bbox_inches='tight')
    plt.close()
    print('Plotted to ' + outFig)


def MhtPlot(dfOutPath, outPath):
    files = RecurReadFilePaths(dfOutPath, files=[])
    for file in files:
        outFig = re.sub(dfOutPath, outPath, file.rstrip('.df.out')) + '.png'
        outFigPath = '/'.join(outFig.split('/')[:-1])
        os.makedirs(outFigPath, exist_ok=True)
        fileName = file.split('/')[-1]
        favPos = int(fileName.split('_')[0])
        df = pd.read_csv(file, sep='\t')
        df.sort_values(by='DF', inplace=True, ascending=False, ignore_index=True)
        try:
            favPosRank = df[df.POS == favPos].index.values[0] + 1
            favScore = df[df.POS == favPos].DF.values[0]
        except:
            warnings.warn('favored mutation site not found in ' + file, UserWarning)
            continue
        regionLen = str(round((df.POS.max() - df.POS.min()) / 1000))
        posList = df.POS.values.tolist()
        scoreList = df.DF.values.tolist()
        fig_size = [8, 4]
        fontsize = 15
        fontsize_tick_params = fontsize - 3
        plt.clf()
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=fig_size, squeeze=True)
        y_lim_below, y_lim_upper = -0.02, 1.1
        ax.set_xlabel('Position(bp)', fontsize=fontsize)
        ax.set_ylabel('DFscore', fontsize=fontsize)
        ax.set_ylim(bottom=y_lim_below, top=y_lim_upper)  # , fontsize=fontsize)
        ax.scatter(posList, scoreList, 50, 'gray')
        ax.scatter(favPos, favScore, 180, 'red', marker='v')
        ax.set_title('favored site rank=%s, region length=%sKb' % (str(favPosRank), regionLen), fontsize=fontsize)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=fontsize_tick_params)

        # Save fig
        plt.xlim()
        plt.savefig(outFig, dpi=200, bbox_inches='tight')
        print('Output to:\n' + outFig)
        plt.close()


def PlotRankCDFcurve(dfOutPath, outFig):
    print('plot rankCDF curve ...')
    dfOutFiles = RecurReadFilePaths(dfOutPath, files=[])
    ranks = []
    regionLens = []
    for file in dfOutFiles:
        fileName = file.split('/')[-1]
        if fileName[:5] == 'sweep':
            favPos = int(fileName.split('_')[1])
            df = pd.read_csv(file, sep='\t')
            df.sort_values(by='DF', inplace=True, ascending=False, ignore_index=True)
            try:
                rank = df[df.POS == favPos].index.values[0] + 1
                ranks.append(rank)
                regionLens.append(df.POS.max() - df.POS.min())
            except:
                pass
    mean_region_len = np.mean(regionLens)
    plt.clf()
    fig_size = [8, 10]
    fontsize = 15
    fontsize_tick_params = fontsize - 3
    line_width = 5
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    counts, cdf_bins = np.histogram(ranks, bins=max(ranks), density=True)
    cdf = np.cumsum(counts)
    ax.plot(cdf_bins[:-1], cdf, color='black', linewidth=line_width)
    ax.locator_params(nbins=5)
    if mean_region_len < 5000:
        xlim_right = 5
    elif mean_region_len < 10000:
        xlim_right = 10
    else:
        xlim_right = 50
    ax.set_xlim(left=1, right=xlim_right)  # , fontsize=fontsize)
    ax.set_ylim(bottom=0, top=1)  # , fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize_tick_params)
    ax.set_title("mean regions length=%iKb, regions number=%i" % (round(mean_region_len / 1000), len(regionLens)),
                 fontsize=fontsize)
    plt.ylabel('Cumulative Distribution Function', fontsize=fontsize)
    plt.xlabel('Rank', fontsize=fontsize)
    plt.savefig(outFig, dpi=200, bbox_inches='tight')
    plt.close()
    print('plotted to ' + outFig)

# }
# ====================================================================================================
