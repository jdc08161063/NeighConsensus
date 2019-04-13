import sys 
sys.path.append('./model')

import argparse 
import torch
import numpy as np
from model.model import NCNet
import torchvision.transforms as transforms
import json 
import os 
import pandas as pd
from evalPascalPckDataloader import resizeImg, getIndexMatchGT
from tqdm import tqdm
import torch.nn.functional as F

from scipy import interpolate


def Bilinear(x, y, x1, y1, x2, y2, z11, z12, z21, z22) : 

    return (z11 * (x2 - x) * (y2 - y) +
            z21 * (x - x1) * (y2 - y) +
            z12 * (x2 - x) * (y - y1) +
            z22 * (x - x1) * (y - y1))/((x2 - x1) * (y2 - y1))    
            
def bilinearInterpolation(xB, yB, posXB, posYB, posXA, posYA, Score) : 

    nbPoint = len(xB)
    diffXB = xB.reshape((-1, 1)) - posXB.reshape((1, -1))
    diffYB = yB.reshape((-1, 1)) - posYB.reshape((1, -1))
    diff = (diffXB ** 2 + diffYB ** 2) 
    
    tmpDiff = np.copy(diff)
    tmpDiff[(diffXB > 0) | (diffYB > 0)] = 2
    rightBottomIndex =  np.argmin(tmpDiff, axis=1)
    rightBottomDiff =  np.min(tmpDiff, axis=1)
    
    tmpDiff = np.copy(diff)
    tmpDiff[(diffXB > 0) | (diffYB < 0)] = 2
    rightTopIndex =  np.argmin(tmpDiff, axis=1)
    rightTopDiff =  np.min(tmpDiff, axis=1)
    
    tmpDiff = np.copy(diff)
    tmpDiff[(diffXB < 0) | (diffYB > 0)] = 2
    leftBottomIndex =  np.argmin(tmpDiff, axis=1)
    leftBottomDiff =  np.min(tmpDiff, axis=1)
    
    
    tmpDiff = np.copy(diff)
    tmpDiff[(diffXB < 0) | (diffYB < 0)] = 2
    leftTopIndex =  np.argmin(tmpDiff, axis=1)
    leftTopDiff =  np.min(tmpDiff, axis=1)
    
    wrapPointxA = np.zeros(len(xB))
    wrapPointyA = np.zeros(len(xB))
    wrapScore = np.zeros(len(xB))
    
    for i in range(nbPoint) : 
        ## if 4 points are found, biliear interpolation see : https://en.wikipedia.org/wiki/Bilinear_interpolation
        if rightBottomDiff[i] != 2 and rightTopDiff[i] != 2 and leftBottomDiff[i] != 2 and leftTopDiff[i] != 2 : 
            
            x, y, x1, y1, x2, y2 = xB[i], yB[i], posXB[rightBottomIndex[i]], posYB[rightBottomIndex[i]], posXB[leftTopIndex[i]], posYB[leftTopIndex[i]]
            
            
            
            z11, z12, z21, z22 = posXA[rightBottomIndex[i]], posXA[rightTopIndex[i]], posXA[leftTopIndex[i]], posXA[leftTopIndex[i]]
            wrapPointxA[i] = Bilinear(x, y, x1, y1, x2, y2, z11, z12, z21, z22)
            
            z11, z12, z21, z22 = posYA[rightBottomIndex[i]], posYA[rightTopIndex[i]], posYA[leftTopIndex[i]], posYA[leftTopIndex[i]]
            wrapPointyA[i] = Bilinear(x, y, x1, y1, x2, y2, z11, z12, z21, z22)
            
            z11, z12, z21, z22 = Score[rightBottomIndex[i]], Score[rightTopIndex[i]], Score[leftTopIndex[i]], Score[leftTopIndex[i]]
            wrapScore[i] = Bilinear(x, y, x1, y1, x2, y2, z11, z12, z21, z22)
        ## OtherWise find the Nearest Neighbour
        else : 
            tmpDiff = [rightBottomDiff[i], rightTopDiff[i], leftBottomDiff[i], leftTopDiff[i]]
            tmpIndex = [rightBottomIndex[i], rightTopIndex[i], leftBottomIndex[i], leftTopIndex[i]]
            NNIndex = np.argmin(tmpDiff)
            wrapPointxA[i], wrapPointyA[i], wrapScore[i] = posXA[tmpIndex[NNIndex]], posYA[tmpIndex[NNIndex]], Score[tmpIndex[NNIndex]]
            
    return  wrapPointxA, wrapPointyA, wrapScore

        
        
    
def evalPascal(model, testTransform, df, featExtractor, imgDir, maxSize, alpha = 0.1, saveDir = None, inter2dType = 'linear') : 

    if featExtractor == 'ResNet18Conv4' : 
        minNet = 15
        strideNet = 16
    elif featExtractor == 'ResNet18Conv5' : 
        minNet = 31
        strideNet = 32
    elif featExtractor == 'ResNet101Conv4' : 
        minNet = 17
        strideNet = 16
    
    
    if saveDir and not os.path.exists(saveDir) : 
        os.mkdir(saveDir)
        
    nbPair = len(df)
    pckRes = np.zeros(nbPair)

    for i in tqdm(range(nbPair)) : 
        IA, IB, xA, yA, xB, yB, refPCK, wAOrg, hAOrg, wBOrg, hBOrg = getIndexMatchGT(df, imgDir, minNet, strideNet, maxSize, i)
        refPCK = 224.
        
        ## Plot GT matching
        if saveDir: 
            outPath = os.path.join(saveDir, 'GT{:d}.jpg'.format(i))
            plotCorres(IA, IB, xA, yA, xB, yB, score = [], scoreTH = 0.5, lineColor = 'green', saveFig = outPath)
        
        ## compute 4d correlation
        featA, featB = testTransform(IA), testTransform(IB)
        featA, featB = featA.unsqueeze(0).cuda(), featB.unsqueeze(0).cuda() 
        with torch.no_grad() : 
            corr4D = model(featA, featB)
        _, _, wA, hA, wB, hB = corr4D.size()
        corr4D = corr4D.view(wA * hA, wB * hB)
        
        ## compute mutual matching
        
        #corr4D = F.softmax(corr4D, dim=0) * F.softmax(corr4D, dim=1)
        #scoreA, posA = corr4D.topk(k=1, dim = 1)
        scoreB, posB = corr4D.topk(k=1, dim = 0)
        #top1ScoreA = torch.cuda.FloatTensor(wA * hA, wB * hB).fill_(0).scatter_(1, posA, scoreA)
        top1ScoreB = torch.cuda.FloatTensor(wA * hA, wB * hB).fill_(0).scatter_(0, posB, scoreB)

        #top1ScoreMM = (top1ScoreA * top1ScoreB) ** 0.5
        top1ScoreMM = top1ScoreB
        indexMM = top1ScoreMM.nonzero()
        scoreMM = torch.masked_select(top1ScoreMM, top1ScoreMM > 0)
        
        
        ## Put value in CPU and compute reference grid matching
        indexMM, scoreMM = indexMM.cpu().numpy(), scoreMM.cpu().numpy()
        posYA, posXA, posYB, posXB  = ((indexMM[:, 0] / hA).astype(int) + 0.5) / float(wA), (indexMM[:, 0] % hA + 0.5) / float(hA), ((indexMM[:, 1] / hB).astype(int) + 0.5) / float(wB), (indexMM[:, 1] % hB + 0.5) / float(hB)
        
        scoreMMSort = np.sort(scoreMM)[::-1]
        nbMatch = len(scoreMMSort)
        
        
        
        
        ## If no mutual matching, precision is 0
        if len(posXA) == 0 :
            continue
            
        ## Plot Grid matching
        if saveDir and len(posXA) > 0: 
            outPath = os.path.join(saveDir, 'Top5_{:d}.jpg'.format(i))
            plotCorres(IA, IB, posXA, posYA, posXB, posYB, scoreMM, scoreMMSort[int(nbMatch * 0.05)], lineColor = 'blue', saveFig = outPath)
            outPath = os.path.join(saveDir, 'Top10_{:d}.jpg'.format(i))
            plotCorres(IA, IB, posXA, posYA, posXB, posYB, scoreMM, scoreMMSort[int(nbMatch * 0.1)], lineColor = 'blue', saveFig = outPath)
            outPath = os.path.join(saveDir, 'Top20_{:d}.jpg'.format(i))
            plotCorres(IA, IB, posXA, posYA, posXB, posYB, scoreMM, scoreMMSort[int(nbMatch * 0.2)], lineColor = 'blue', saveFig = outPath)
            
        ## interpolation
        wrapPointxA, wrapPointyA, wrapScore = bilinearInterpolation(xB, yB, posXB, posYB, posXA, posYA, scoreMM)
        
        if saveDir : 
            outPath = os.path.join(saveDir, 'KeyPointMatch{:d}.jpg'.format(i))
            plotCorres(IA, IB, wrapPointxA, wrapPointyA, xB, yB, lineColor = 'blue', saveFig = outPath)
        
        ## PCK Metric
       
        prec = np.mean(((wrapPointxA * wAOrg - xA * wAOrg) ** 2 + (wrapPointyA * hAOrg - yA * hAOrg) ** 2) ** 0.5 <= refPCK * alpha)
        pckRes[i] = prec
        print (prec)
    return pckRes



if __name__ == '__main__' : 

    ## Parameters
    parser = argparse.ArgumentParser(description='Nc-Net Training')

    ## Input / Output 
    parser.add_argument('--saveDir', type=str, help='output visual directory')
    parser.add_argument('--resumePth', type=str, help='resume model path')
    parser.add_argument('--imgDir', type=str, default = 'data/pf-pascal/JPEGImages/', help='image Directory')
    parser.add_argument('--testCSV', type=str, default = 'data/pf-pascal/test.csv', help='train csv')
    parser.add_argument('--maxSize', type=int, default = 500, help='max size in the test image')


    ## evaluation parameters
    parser.add_argument('--neighConsKernel', nargs='+', type=int, default=[5,5,5], help='kernels sizes in neigh. cons.')
    parser.add_argument('--neighConsChannel', nargs='+', type=int, default=[16,16,1], help='channels in neigh. cons')
    parser.add_argument('--featExtractor', type=str, default='ResNet101Conv4', choices=['ResNet18Conv4', 'ResNet18Conv5', 'ResNet101Conv4'], help='feature extractor')
    parser.add_argument('--inter2dType', type=str, default='linear', choices=['linear', 'cubic', 'quintic'], help='number of point to do interpolation')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha in the PCK metric')


    args = parser.parse_args()
    print(args)
    ## Initial Model
    model = NCNet(kernel_sizes=args.neighConsKernel, 
                  channels=args.neighConsChannel, 
                  featExtractor = args.featExtractor, 
                  finetuneFeatExtractor = False)
            


    msg = '\nResume from {}'.format(args.resumePth)
    print (msg)
    modelParam = torch.load(args.resumePth)
    modelParam = modelParam['state_dict']
    for key in list(modelParam.keys()) : 
        modelParam[key.replace('FeatureExtraction.model', 'featExtractor').replace('NeighConsensus.conv', 'NeighConsensus.model')] = modelParam[key]
    model.load_state_dict(modelParam, strict=False)

    #model.load_state_dict(torch.load(args.resumePth))
        
    model.cuda()
    model.eval()
        

    ## Train Val DataLoader
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
    testTransform = transforms.Compose([transforms.ToTensor(), 
                                         normalize,])

    
        
        
    # pck results
    df = pd.read_csv(args.testCSV)
    pckRes = evalPascal(model, testTransform, df, args.featExtractor, args.imgDir, args.maxSize, args.alpha, args.saveDir, args.inter2dType)
    msg = 'NB Valid Pairs {:d}, PCK : {:.4f}'.format( np.sum(pckRes >= 0),  np.sum(pckRes * (pckRes >= 0).astype(np.float)) / np.sum(pckRes >= 0) )
    print (msg)
        







