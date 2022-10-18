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
        self.transform = transforms_dict['transform_gt_train']

    def extract_feats(self, x, use_mtcnn):
        if use_mtcnn:
            x = tensor2im(x[0])
            x, _ = self.mtcnn.align(x)
            x = self.transform(x)
            x = x.unsqueeze(0).cuda()
        else:
            x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)

        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x, use_mtcnn):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x, use_mtcnn)
        y_feats = self.extract_feats(y, use_mtcnn)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat, use_mtcnn)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({'diff_target': float(diff_target),
                            'diff_input': float(diff_input),
                            'diff_views': float(diff_views)})
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs
