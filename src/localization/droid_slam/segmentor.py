from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights, lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights, deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torch
import torch.nn as nn

class SemSegNet(nn.Module):
    """
        network_name: Name of pre-trained segmentation network (either FCN, DeepLabV3 or LRASPP)
        visualize: whether to save an image of the input without the mask
    """
    def __init__(self, network_name, visualize=False):
        super(SemSegNet, self).__init__()
        assert(network_name in ["FCN", "DeepLabV3", "LRASPP"])

        if network_name == "FCN":
            self.weights = FCN_ResNet50_Weights.DEFAULT
            self.model = fcn_resnet50(weights=self.weights)
        elif network_name == "DeepLabV3":
            self.weights = DeepLabV3_ResNet50_Weights.DEFAULT
            self.model = deeplabv3_resnet50(weights=self.weights)
        else:
            self.weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
            self.model = lraspp_mobilenet_v3_large(weights=self.weights)
            
        self.model.eval()

        self.preprocess = self.weights.transforms()

        self.visualize = visualize

    def forward(self, image, device):
        """
            image: torch.tensor of shape (3, H, W)
            depth: torch.tensor of shape (3, H, W)
        """
        with torch.no_grad():
            self.model.to(device)
            preproc_image = self.preprocess(image).unsqueeze(dim=0)
            prediction = self.model(preproc_image)["out"]
            normalized_masks = prediction.softmax(dim=1)

            class_to_idx = {cls: idx for (idx, cls) in enumerate(self.weights.meta["categories"])}

            masks = torch.nn.functional.interpolate(normalized_masks, size=image.shape[1:], mode='bilinear')

            class_dim = 1
            boolean_person_masks = masks.argmax(class_dim) == class_to_idx['person']
            boolean_dog_masks = masks.argmax(class_dim) == class_to_idx['dog']
            boolean_cat_masks = masks.argmax(class_dim) == class_to_idx['cat']
            boolean_masks = boolean_person_masks | boolean_dog_masks | boolean_cat_masks

        return boolean_masks