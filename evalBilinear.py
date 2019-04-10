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

def getPixelValueFromIndex(featExtractor, I, indexW, indexH) : 
    if featExtractor == 'ResNet18Conv4' : 
        minNet = 15
        strideNet = 16
    elif featExtractor == 'ResNet18Conv5' : 
        minNet = 31
        strideNet = 32
    elif featExtractor == 'ResNet101Conv4' : 
        minNet = 17
        strideNet = 16
        w, h = I.size
        posW = (indexW - 2) * strideNet + minNet + strideNet / 2 if indexW >= 2 else minNet / 2 * (indexW + 0.5)
        posH = (indexH - 2) * strideNet + minNet + strideNet / 2 if indexH >= 2 else minNet / 2 * (indexH + 0.5)
        #print (w, h, indexW, indexH, float(posW / w), float(posH / h))
        return float(posW / w), float(posH / h)
        

def evalPascal(model, testTransform, df, featExtractor, softmaxMM, imgDir, maxSize, scoreTH, sigma, nbWeightPoint, alpha = 0.1, saveDir = None) : 

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
        #refPCK = 224.
        
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
        #if not softmaxMM : 
        #    corr4D = F.softmax(corr4D, dim = 0) * F.softmax(corr4D, dim = 1)
        
        ## compute mutual matching
        
        scoreA, posA = corr4D.topk(k=1, dim = 1)
        scoreB, posB = corr4D.topk(k=1, dim = 0)
        top1ScoreA = torch.cuda.FloatTensor(wA * hA, wB * hB).fill_(0).scatter_(1, posA, scoreA)
        top1ScoreB = torch.cuda.FloatTensor(wA * hA, wB * hB).fill_(0).scatter_(0, posB, scoreB)

        top1ScoreMM = (top1ScoreA * top1ScoreB) ** 0.5
        indexMM = top1ScoreMM.nonzero()
        scoreMM = torch.masked_select(top1ScoreMM, top1ScoreMM > 0)
        '''
        scoreB, posB = corr4D.topk(k=1, dim = 0)
        top1ScoreB = torch.cuda.FloatTensor(wA * hA, wB * hB).fill_(0).scatter_(0, posB, scoreB)
        top1ScoreMM = top1ScoreB
        indexMM = top1ScoreMM.nonzero()
        scoreMM = torch.masked_select(top1ScoreMM, top1ScoreMM > 0)
        '''
        
        ## Put value in CPU and compute reference grid matching
        indexMM, scoreMM = indexMM.cpu().numpy(), scoreMM.cpu().numpy()
        posYA, posXA, posYB, posXB  = ((indexMM[:, 0] / hA).astype(int) + 0.5) / float(wA), (indexMM[:, 0] % hA + 0.5) / float(hA), ((indexMM[:, 1] / hB).astype(int) + 0.5) / float(wB), (indexMM[:, 1] % hB + 0.5) / float(hB)
        #posYA, posXA, posYB, posXB  = ((indexMM[:, 0] / hA).astype(int) ) / float(wA - 1), (indexMM[:, 0] % hA) / float(hA - 1), ((indexMM[:, 1] / hB).astype(int) ) / float(wB - 1), (indexMM[:, 1] % hB ) / float(hB - 1)
        '''
        posYA, posXA, posYB, posXB = (indexMM[:, 0] / hA).astype(int).astype(float), (indexMM[:, 0] % hA).astype(float), (indexMM[:, 1] / hB).astype(int).astype(float), (indexMM[:, 1] % hB).astype(float)
        
        for j in range(len(indexMM)) : 
            tmpXA, tmpYA = getPixelValueFromIndex(featExtractor, IA, posXA[j], posYA[j])
            posXA[j], posYA[j] = tmpXA, tmpYA
            
            
            tmpXB, tmpYB = getPixelValueFromIndex(featExtractor, IB, posXB[j], posYB[j])
            posXB[j], posYB[j] = tmpXB, tmpYB'''
        
        ## If no mutual matching, precision is 0
        if len(posXA) == 0 :
            continue
            
        ## Plot Grid matching
        if saveDir and len(posXA) > 0: 
            outPath = os.path.join(saveDir, 'Grid{:d}.jpg'.format(i))
            plotCorres(IA, IB, posXA, posYA, posXB, posYB, scoreMM, scoreTH, lineColor = 'blue', saveFig = outPath)
            
        ## compute weight 
        diffXB = xB.reshape((-1, 1)) - posXB.reshape((1, -1))
        diffYB = yB.reshape((-1, 1)) - posYB.reshape((1, -1))
        
        diff = (diffXB ** 2 + diffYB ** 2) 
        diff[(diffXB > 0) | (diffYB > 0)] = 2
        rightBottom =  np.argmin(diff, axis=1)
        

        diff = (diffXB ** 2 + diffYB ** 2) 
        diff[(diffXB > 0) | (diffYB < 0)] = 2
        rightTop =  np.argmin(diff, axis=1)
        
        diff = (diffXB ** 2 + diffYB ** 2) 
        diff[(diffXB < 0) | (diffYB > 0)] = 2
        leftBottom =  np.argmin(diff, axis=1)
        

        diff = (diffXB ** 2 + diffYB ** 2) 
        diff[(diffXB < 0) | (diffYB < 0)] = 2
        leftTop =  np.argmin(diff, axis=1)
        

        wrapPointxA = np.zeros(len(xA))
        wrapPointyA = np.zeros(len(xA))
        wrapScore = np.zeros(len(xA))
        diff = (diffXB ** 2 + diffYB ** 2) 

        for j in range(len(xA)) : 
            
            weight = np.exp(np.array([diff[j, leftTop[j]], diff[j, leftBottom[j]], diff[j, rightTop[j]], diff[j, rightBottom[j]]]) * -1 / sigma / 2)
            interPosXA = np.array([posXA[leftTop[j]], posXA[leftBottom[j]], posXA[rightTop[j]], posXA[rightBottom[j]]])
            interPosYA = np.array([posYA[leftTop[j]], posYA[leftBottom[j]], posYA[rightTop[j]], posYA[rightBottom[j]]])
        
            wrapPointxA[j] = (interPosXA * weight).sum() / weight.sum()
            wrapPointyA[j] = (interPosYA * weight).sum() / weight.sum()

        if saveDir : 
            outPath = os.path.join(saveDir, 'KeyPointMatch{:d}.jpg'.format(i))
            plotCorres(IA, IB, wrapPointxA, wrapPointyA, xB, yB, lineColor = 'blue')
        
        ## PCK Metric
       
        prec = np.mean(((wrapPointxA * wAOrg - xA * wAOrg) ** 2 + (wrapPointyA * hAOrg - yA * hAOrg) ** 2) ** 0.5 <= refPCK * alpha)
        pckRes[i] = prec
        #print (prec)
        #break
    return pckRes



if __name__ == '__main__' : 

    ## Parameters
    parser = argparse.ArgumentParser(description='Nc-Net Training')

    ## Input / Output 
    parser.add_argument('--saveDir', type=str, help='output visual directory')
    parser.add_argument('--resumePth', type=str, help='resume model path')
    parser.add_argument('--featExtractorPth', type=str, default = 'model/FeatureExtractor/resnet18.pth', help='feature extractor path')
    parser.add_argument('--imgDir', type=str, default = 'data/pf-pascal/JPEGImages/', help='image Directory')
    parser.add_argument('--testCSV', type=str, default = 'data/pf-pascal/test.csv', help='train csv')
    parser.add_argument('--maxSize', type=int, default = 1000, help='max size in the test image')


    ## evaluation parameters
    parser.add_argument('--neighConsKernel', nargs='+', type=int, default=[5,5,5], help='kernels sizes in neigh. cons.')
    parser.add_argument('--neighConsChannel', nargs='+', type=int, default=[16,16,1], help='channels in neigh. cons')
    parser.add_argument('--featExtractor', type=str, default='ResNet18Conv4', choices=['ResNet18Conv4', 'ResNet18Conv5', 'ResNet101Conv4'], help='feature extractor')
    parser.add_argument('--softmaxMM', action='store_true', help='whether use softmax Mutual Matching')
    parser.add_argument('--scoreTH', type=float, default=0.5, help='threshold of score to visualize matched grid')
    parser.add_argument('--sigma', type=float, default=0.01, help='sigma in the spatial term to compute weight')
    parser.add_argument('--nbWeightPoint', type=int, default=4, help='number of point to do interpolation')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha in the PCK metric')


    args = parser.parse_args()
    print(args)
    ## Initial Model
    model = NCNet(kernel_sizes=args.neighConsKernel, 
                  channels=args.neighConsChannel, 
                  featExtractor = args.featExtractor, 
                  featExtractorPth = args.featExtractorPth, 
                  finetuneFeatExtractor = False,
                  softmaxMutualMatching = args.softmaxMM)
            


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
    pckRes = evalPascal(model, testTransform, df, args.featExtractor, args.softmaxMM, args.imgDir, args.maxSize, args.scoreTH, args.sigma, args.nbWeightPoint, args.alpha, args.saveDir)
    msg = 'NB Valid Pairs {:d}, PCK : {:.4f}'.format( np.sum(pckRes >= 0),  np.sum(pckRes * (pckRes >= 0).astype(np.float)) / np.sum(pckRes >= 0) )
    print (msg)
        







