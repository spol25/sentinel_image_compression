from .detection_model import DetectionFramework
from .classifier_model import ClassifierFramework
from .maskgit_model import MaskGITFramework, ResNetTokenizerFramework

# Basheim handling cases in the intrim while waiting for AI team's direction
class FrameworkFactory:

    @staticmethod
    def get_framework(config):
        framework = config.get("framework")
        if framework == "YOLOv5":
            return DetectionFramework("YOLOv5", config)
        elif framework == "2StageClassifier":
            return ClassifierFramework("2StageClassifier", config)
        elif framework == "MaskGIT":
            return MaskGITFramework("MaskGIT", config)
        elif framework == "ResNetTokenizer":
            return ResNetTokenizerFramework("ResNetTokenizer", config)
        else:
            raise Exception("Framework can not be found")
