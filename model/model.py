import torch 
from neighConsensus import NeighConsensus, MutualMatching, MutualMatchingSoftMax
from util import featureL2Norm, featureCorrelation
import torchvision.models as models
from torch import nn

class NCNet(torch.nn.Module):
    def __init__(self, kernel_sizes, channels, featExtractor, finetuneFeatExtractor):
        super(NCNet, self).__init__()
        
        ## Define feature extractor
        resnet_feature_layers = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4']
        if featExtractor == 'ResNet18Conv4' : 
            print ('Loading ResNet 18 Conv4 Weight ...')
            self.featExtractor = models.resnet18(pretrained=True)            
            resnet_module_list = [getattr(self.featExtractor,l) for l in resnet_feature_layers]
            last_layer_idx = resnet_feature_layers.index('layer3')
            self.featExtractor = nn.Sequential(*resnet_module_list[:last_layer_idx+1])

        elif featExtractor=='ResNet101Conv4':
            print ('Loading ResNet 101 Conv4 Weight ...')
            self.featExtractor = models.resnet101(pretrained=True)            
            resnet_module_list = [getattr(self.featExtractor,l) for l in resnet_feature_layers]
            last_layer_idx = resnet_feature_layers.index('layer3')
            self.featExtractor = nn.Sequential(*resnet_module_list[:last_layer_idx+1])
            
        self.finetuneFeatExtractor = finetuneFeatExtractor 
        

            
        if not self.finetuneFeatExtractor : 
            msg = '\nSet Feature Extractor to validation mode...'
            print (msg)
            self.featExtractor.eval()

         ## Mutual Matching method
        self.mutualMatch = MutualMatching
        
        ## NeighConsensus
        self.NeighConsensus = NeighConsensus(kernel_sizes, channels)
        
        

    def forward(self, xA, xB):
        
        ## Extract Feature
        if not self.finetuneFeatExtractor : 
            with torch.no_grad() : 
                featA, featB = self.featExtractor(xA), self.featExtractor(xB)
        ## Normalization
                featANorm, featBNorm = featureL2Norm(featA), featureL2Norm(featB)
        ## Correlation Tensor
                corr4d = featureCorrelation(featANorm, featBNorm)
        
        else : 
        
            featA, featB = self.featExtractor(xA), self.featExtractor(xB)
            featANorm, featBNorm = featureL2Norm(featA), featureL2Norm(featB)
            corr4d = featureCorrelation(featANorm, featBNorm)
        
        ## Mutual Match 
        corr4d = self.mutualMatch(corr4d)
        ## Neighbor Consensus
        corr4d = self.NeighConsensus(corr4d)
        ## Mutual Match 
        corr4d = self.mutualMatch(corr4d)
        
        return corr4d
        
        
if __name__ == '__main__' : 
    print ('Test NcNet...')
    model = NCNet(kernel_sizes=[5,5,5], channels=[16,16,1], featExtractor = 'ResNet18Conv4')
    a = torch.randn(1, 3, 224, 224).cuda()
    b = torch.randn(1, 3, 224, 224).cuda()
    
    model.cuda()
    
    print (model(a, b).size()) # output is 1, 1, 11, 12, 13, 14
   
