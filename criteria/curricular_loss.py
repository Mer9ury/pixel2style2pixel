import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone
from models.encoders.model_irse import IR_101
from models.mtcnn.mtcnn import MTCNN
from configs import data_configs
from utils.common import tensor2im



class IDLoss(nn.Module):
    def __init__(self,rank = 0,use_curricular = False):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        if use_curricular:
            self.facenet = IR_101(input_size=112)
            self.facenet.load_state_dict(torch.load(model_paths['circular_face'],map_location=torch.device(rank)))
        else:
            self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').cuda(rank)
            self.facenet.load_state_dict(torch.load(model_paths['ir_se50'],map_location=torch.device(rank)))
        
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.mtcnn = MTCNN()

        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False

        self.facenet.cuda()
        self.facenet.eval()

        dataset_args = data_configs.DATASETS['ffhq_encode']
        transforms_dict = dataset_args['transforms'](-1).get_transforms()
        self.transform = transforms_dict['transform_id_loss']

    def extract_feats(self, x, use_mtcnn):
        if use_mtcnn:
            # x = tensor2im(x[0])
            x, _ = self.mtcnn.align(x)
            x = self.transform(x)
            x = x.unsqueeze(0).cuda()
            
        else:
            x = x[:, :, 35:223, 32:220]  # Crop interesting region
            x = self.face_pool(x)

        x_feats = self.facenet(x)
        return x_feats[0]

    def forward(self, y, x, use_mtcnn=False):

        x_feats = self.extract_feats(x, use_mtcnn)
        y_feats = self.extract_feats(y, use_mtcnn)  # Otherwise use the feature from there

        diff_target = x_feats.dot(y_feats)

        return diff_target.item()
    