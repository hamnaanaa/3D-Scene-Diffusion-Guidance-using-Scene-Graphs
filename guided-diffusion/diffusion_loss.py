import torch
import torch.nn as nn

class DiffusionLoss(nn.Module):
    # 1e-5 seemed to offer a good balance for the losses
    def __init__(self, weight_l2=1.0, weight_volume=0, weight_aspect=0, weight_location=0, weight_interdistance=0):
        super(DiffusionLoss, self).__init__()
        self.weight_l2 = weight_l2
        self.weight_volume = weight_volume
        self.weight_aspect = weight_aspect
        self.weight_location = weight_location
        self.weight_interdistance = weight_interdistance

    def forward(self, predicted, ground_truth):
        # General L2 loss
        if self.weight_l2 > 0:
            l2_loss = nn.MSELoss()(predicted, ground_truth) * self.weight_l2
        else:
            l2_loss = torch.tensor(0.0, device=predicted.device)

        # Volume difference of bounding boxes loss (size is encoded as 3 length values in channels 12-14)
        if self.weight_volume > 0:
            predicted_volume = predicted[..., 12] * predicted[..., 13] * predicted[..., 14]
            gt_volume = ground_truth[..., 12] * ground_truth[..., 13] * ground_truth[..., 14]
            volume_loss = nn.MSELoss()(predicted_volume, gt_volume) * self.weight_volume
        else:
            volume_loss = torch.tensor(0.0, device=predicted.device)

        # Aspect ratio loss (size is encoded as 3 length values in channels 12-14)
        if self.weight_aspect > 0:
            predicted_aspect = self._get_exp_aspect_ratios_tensor(predicted)
            gt_aspect = self._get_exp_aspect_ratios_tensor(ground_truth)
            aspect_loss = nn.MSELoss()(predicted_aspect, gt_aspect) * self.weight_aspect
            # clamp to avoid huge loss values when the aspect ratio is close to zero
            aspect_loss = torch.clamp(aspect_loss, min=0, max=1.25)
        else:
            aspect_loss = torch.tensor(0.0, device=predicted.device)
        
        # Pairwise location between centroids loss
        if self.weight_location > 0:
            pairwise_distances = self._get_pairwise_distances_tensor(predicted, ground_truth)
            location_loss = torch.sum(pairwise_distances) * self.weight_location
        else:
            location_loss = torch.tensor(0.0, device=predicted.device)
            
        # Pairwise inter-distance between centroids loss
        if self.weight_interdistance > 0:
            predicted_pairwise_distances, gt_pairwise_distances = self._get_interdistance_relationship(predicted, ground_truth)
            interdistance_loss = nn.MSELoss()(predicted_pairwise_distances, gt_pairwise_distances) * self.weight_interdistance
        else:
            interdistance_loss = torch.tensor(0.0, device=predicted.device)


        total_loss = l2_loss + volume_loss + aspect_loss + location_loss + interdistance_loss

        return (
            # global loss
            total_loss, # / (5), # self.weight_l2 + self.weight_volume + self.weight_aspect + self.weight_location + self.weight_interdistance
            # individual losses
            l2_loss, volume_loss, aspect_loss, location_loss, interdistance_loss
        )
    
    def _get_exp_aspect_ratios_tensor(self, tensor):
        # add epsilon to avoid division by zero for numerical stability
        
        # exp_aspect1 = torch.exp(tensor[..., 12]) / torch.exp(tensor[..., 13])
        # exp_aspect2 = torch.exp(tensor[..., 12]) / torch.exp(tensor[..., 14])
        # exp_aspect3 = torch.exp(tensor[..., 13]) / torch.exp(tensor[..., 14])
        
        eps = 1e-4 + 50
        # to avoid numerical instability, we use the log of the length values
        exp_aspect1 = torch.log(tensor[..., 12] + eps) - torch.log(tensor[..., 13] + eps)
        exp_aspect2 = torch.log(tensor[..., 12] + eps) - torch.log(tensor[..., 14] + eps)
        exp_aspect3 = torch.log(tensor[..., 13] + eps) - torch.log(tensor[..., 14] + eps)
        exp_aspect_ratios = torch.stack([exp_aspect1, exp_aspect2, exp_aspect3], dim=-1)
        return exp_aspect_ratios
    
    def _get_pairwise_distances_tensor(self, tensor1, tensor2):
        # Pairwise distances between centroids loss (centroids are in the first three channels)
        predicted_centroids = tensor1[..., :3]
        gt_centroids = tensor2[..., :3]

        # Compute pairwise distances between centroids using broadcasting
        predicted_centroids_expand = predicted_centroids.unsqueeze(1)  # [B, 1, N, 3]
        gt_centroids_expand = gt_centroids.unsqueeze(2)  # [B, N, 1, 3]
        pairwise_distances = torch.norm(predicted_centroids_expand - gt_centroids_expand, dim=-1)  # [B, N, N]
        return pairwise_distances
    
    def _get_interdistance_relationship(self, tensor1, tensor2):
        # Pairwise distances between centroids loss (centroids are in the first three channels)
        predicted_centroids = tensor1[..., :3]
        gt_centroids = tensor2[..., :3]

        # Compute pairwise distances between centroids using broadcasting of prediction
        predicted_centroids_expand_1 = predicted_centroids.unsqueeze(1) # [B, 1, N, 3]
        predicted_centroids_expand_2 = predicted_centroids.unsqueeze(2) # [B, N, 1, 3]
        predicted_pairwise_distances = torch.norm(predicted_centroids_expand_1 - predicted_centroids_expand_2, dim=-1)
        
        # Compute pairwise distances between centroids using broadcasting of ground truth
        gt_centroids_expand_1 = gt_centroids.unsqueeze(1) # [B, 1, N, 3]
        gt_centroids_expand_2 = gt_centroids.unsqueeze(2) # [B, N, 1, 3]
        gt_pairwise_distances = torch.norm(gt_centroids_expand_1 - gt_centroids_expand_2, dim=-1)
        
        return predicted_pairwise_distances, gt_pairwise_distances