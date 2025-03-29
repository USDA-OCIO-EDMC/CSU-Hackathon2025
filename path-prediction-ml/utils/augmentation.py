import albumentations as A
import numpy as np

def get_augmentation_transform():
    """Returns an augmentation function using Albumentations."""
    transform = A.Compose([
         A.HorizontalFlip(p=0.5),
         A.VerticalFlip(p=0.5),
         A.Rotate(limit=30, p=0.5),
         A.RandomBrightnessContrast(p=0.5),
    ])
    def augment(image, mask):
         image = image.transpose(1, 2, 0)  # CHW to HWC
         augmented = transform(image=image, mask=mask)
         image_aug = augmented["image"].transpose(2, 0, 1)  # back to CHW
         mask_aug = augmented["mask"]
         return {"image": image_aug, "mask": mask_aug}
    return augment
