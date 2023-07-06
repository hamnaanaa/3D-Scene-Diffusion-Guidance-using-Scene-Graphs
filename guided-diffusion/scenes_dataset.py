import torch
from torch_geometric.data import Data, Dataset

class ScenesDataset(Dataset):
    def __init__(self, scenes):
        super(ScenesDataset, self).__init__()
        self.scenes = scenes

    def len(self):
        return len(self.scenes)
    
    def get(self, index):
        scene = self.scenes[index]
        
        scene_matrix = torch.tensor(scene["scene_matrix"], dtype=torch.float32)
        graph_objects = torch.tensor(scene["graph_objects"], dtype=torch.float32)
        graph_edges = torch.tensor(scene["graph_edges"], dtype=torch.long)
        graph_relationships = torch.tensor(scene["graph_relationships"], dtype=torch.long)

        # Load label strings as well
        labels = scene["labels"]

        return Data(
            x=scene_matrix,
            edge_index=graph_edges,
            edge_attr=graph_relationships,
            cond=graph_objects,
            labels=labels
        )
        

class DatasetConstants:
    """
    Class to hold constants for different datasets used in the paper.
    """
    @staticmethod
    def get_range_matrix():
        """
        Get the range matrix for the 3RScan dataset to normalize the data.
        """

        location_max = torch.tensor([3.285, 3.93, 0.879])
        location_min = torch.tensor([-3.334, -2.619, -1.329])

        normalized_axes_max = torch.ones(9)
        normalized_axes_min = -torch.ones(9)

        size_max = torch.tensor([4.878, 2.655, 2.305])
        size_min = torch.tensor([0.232, 0.14, 0.094])

        range_max = torch.cat((location_max, normalized_axes_max, size_max), dim=0)
        range_min = torch.cat((location_min, normalized_axes_min, size_min), dim=0)

        range_matrix = torch.cat((range_max.unsqueeze(0), range_min.unsqueeze(0)), dim=0)
        
        return range_matrix
