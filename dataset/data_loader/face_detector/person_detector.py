"""Person boxes for the vHRM person_face crop mode: the face is searched inside the largest person of the frame.

torchvision's Faster R-CNN (ResNet-50 FPN v2) trained on COCO; its weights are downloaded once into the torch hub
cache (download_weights, called before the preprocessing workers start so that they do not all download them).
"""

import numpy as np
import torch

MODEL_NAME = "fasterrcnn_resnet50_fpn_v2"
PERSON_LABEL = 1   # COCO category "person"
MIN_SCORE = 0.7    # person detections below this confidence are ignored


def largest_box(boxes, scores, labels, min_score=MIN_SCORE, label=PERSON_LABEL):
    """[x1, y1, x2, y2] (ints) of the largest box of the label scoring at least min_score, or None."""
    candidates = [box for box, score, kind in zip(boxes, scores, labels) if kind == label and score >= min_score]
    if not candidates:
        return None
    largest = max(candidates, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    return [int(round(float(value))) for value in largest]


def download_weights():
    """Put the detector weights into the torch hub cache (no-op when they are there)."""
    import os
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    if not os.path.isfile(os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(weights.url))):
        weights.get_state_dict(progress=True)


class PersonDetector:
    def __init__(self, device=None):
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

        if device is None or (str(device).startswith("cuda") and not torch.cuda.is_available()):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                                box_score_thresh=MIN_SCORE).eval().to(self.device)

    def largest_person(self, frame):
        """Box [x1, y1, x2, y2] of the largest person in an RGB frame (H, W, 3), or None."""
        image = torch.from_numpy(np.ascontiguousarray(frame[..., :3], dtype=np.uint8)).permute(2, 0, 1)
        with torch.no_grad():
            output = self.model([image.to(self.device).float().div_(255.0)])[0]
        box = largest_box(output["boxes"].cpu().numpy(), output["scores"].cpu().numpy(), output["labels"].cpu().numpy())
        if box is None:
            return None
        height, width = frame.shape[:2]
        box = [max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])]
        return box if box[2] > box[0] and box[3] > box[1] else None
