import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoloLoss,self).__init__()
        self.B = B
        self.C = C
        self.S = S

        self.mse = nn.MSELoss(reduction="sum")
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    
    def forward(self,predications, target):
        predications = predications.reshape(-1, self.S, self.S, self.C + (self.B *5))

        iou_1 = intersection_over_union(predications[...,21:25], target[..., 21:25])
        iou_2 = intersection_over_union(predications[...,26:30], target[..., 21:25])
        ious = torch.cat(iou_1.unsqueeze(0),iou_2.unsqueeze(0), dim = 0)
        iou_maxes , bestbox = torch.max(ious, dim=0)  #bestbox will be zero or one 
        exists_box = target[...,20].unsqueeze()  #iobj_i


        # for box coordinates
        box_predictions = exists_box * (
            bestbox * predications[..., 26:30] + (1- bestbox) * predications[...,21:25] 
        )


        box_targets = exists_box * target[..., 21:25]
        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(torch.abs(box_predictions[...,2:4]) + 1e-6)


        box_targets[2:4] = torch.sqrt(box_targets[2:4])
        box_loss = self.mse(torch.flatten(box_predictions, end_dim= -2),
                            torch.flatten(box_targets, end_dim=-2))

    # object loss
        pred_box = (bestbox * box_predictions[...,25:26]) + (1- bestbox) * box_predictions[...,20:21]

        # n*s*s2

        object_loss = self.mse(torch.flatten(exists_box * pred_box), torch.flatten(exists_box * target[...,20:21]))

        # no object loss

        no_objet_loss = self.mse(torch.flatten((1-exists_box) * predications[...,20:21], start_dim=1), torch.flatten((1- exists_box) * target[...,20:21], start_dim=1))

        no_objet_loss+= self.mse(torch.flatten((1-exists_box) * predications[...,25:26], start_dim=1), torch.flatten((1- exists_box) * target[...,20:21], start_dim=1))



        #class loss
        class_loss = self.mse(torch.flatten(exists_box * predications[...,20], end_dim=-2), torch.flatten(exists_box * target[...,20], end_dim=-2))


        total_loss =  self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_objet_loss + class_loss

        return total_loss



