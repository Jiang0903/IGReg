import torch
import torch.nn as nn
import numpy as np
import geotransformer.modules.geotransformer.resunet as resunet




class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        x = 1
        if x == 1:
            self.backbone = resunet.Res50UNet(256, pretrained='imagenet')
        elif x == 2:
            self.backbone = resunet.Res50UNet(256, pretrained=True)
            self.backbone = self.resume_checkpoint('weight/Pri3D_view_geo_ScanNet_ResNet50.pth')

    def resume_checkpoint(self, checkpoint_filename=''):
        import os
        from torch.serialization import default_restore_location
        if os.path.isfile(checkpoint_filename):
            print('===> Loading existing checkpoint')
            state = torch.load(checkpoint_filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            # print(self.backbone2d)
            # load weights
            model = self.backbone
            matched_weights = self.load_state_with_same_shape(model, state['model'])
            # print("matched weight: ",matched_weights)
            model.load_state_dict(matched_weights, strict=False)
            del state
            return model

    def load_state_with_same_shape(self,model, weights):
        # self.logger.write("Loading weights:" + ', '.join(weights.keys()))
        model_state = model.state_dict()
        filtered_weights={}
        for k, v in model_state.items():
            if k in weights and v.size() == model_state[k].size():
                filtered_weights[k]=v
            else:
                filtered_weights[k]=model_state[k]

        # filtered_weights = {
        #     k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
        # }
        # self.logger.write("Loaded weights:" + ', '.join(filtered_weights.keys()))
        return filtered_weights


    def forward(self, x):
        img_feature = self.backbone(x)
        return img_feature