# pylint: disable=[E1101,W0212]

import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class DepthNetwork:
    """
    A wrapper of different depth network (ZoeDepth and Metric3D)
    """
    def __init__(
        self,
        method: str = "ZoeDepth",
        depth_type: str = "ZoeD_NK",
        pretrain: bool = True,
        depth_min: float = 0.1,
        depth_max: float = 1000,
        device: str = "cuda"
    ) -> None:
        self.method = method
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.device = device

        if method == "ZoeDepth":
            self.depth_network = torch.hub.load(
                'isl-org/ZoeDepth',
                depth_type,
                pretrained=pretrain,
            ).to(self.device)
        elif method == "metric3d":
            self.depth_network = torch.hub.load(
                'yvanyin/metric3d',
                depth_type,
                pretrain=pretrain,
            ).to(self.device)
        else:
            raise NotImplementedError
    
    def infer(self, image: torch.Tensor):
        """
        Param:
            @param image: [B,3,H,W]
        Return:
            depth: depth map for image [B,1,H,W]
            confidence: confidence score corresponds to the depth map
            output_dict: other outputs from metric3d
        """
        confidence = None
        output_dict = None
        if self.method == "ZoeDepth":
            depth = self.depth_network.infer(image)
        elif self.method == "metric3d":
            depth, confidence, output_dict = self.depth_network.inference({'input': image})
        else:
            raise NotImplementedError
        
        depth = torch.clamp(depth, self.depth_min, self.depth_max)

        return depth, confidence, output_dict
