import torch
import torchvision
from torchvision.transforms import functional
from PIL import Image


class MinMaxScale:

    def __init__(self):
        pass

    def __call__(self, x):
        pass


class Rotate:
    """Rotate by a given angle."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return functional.rotate(x, self.angle, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)


class RandomRotation:

    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, x):
        angle = (torch.rand(1) * (self.min_angle - self.max_angle) + self.max_angle).item()

        return functional.rotate(
            img=x,
            angle=angle,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )


class RandomScaling:
    """Scale by a given factor."""

    def __init__(self, min_factor, max_factor):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, x):
        """

        @param x: Input image or batch of arbitrary number of leading dimensions.
        @return:
        """
        factor = (torch.rand(1) * (self.min_factor - self.max_factor) + self.max_factor).item()

        width, height = functional._get_image_size(x)

        width_from_center = width // 2
        height_from_center = height // 2

        crop_top = int(height_from_center - height_from_center / factor)
        crop_left = int(width_from_center - width_from_center / factor)

        crop_width = int((width_from_center / factor) * 2)
        crop_height = int((height_from_center / factor) * 2)

        return functional.resized_crop(
            img=x,
            top=crop_top,
            left=crop_left,
            height=crop_height,
            width=crop_width,
            size=[width, height],
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )


class Scale:
    """Scale by a given factor."""

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        """

        @param x: Input image or batch of arbitrary number of leading dimensions.
        @return:
        """
        width_from_center = x.shape[-1] // 2
        height_from_center = x.shape[-2] // 2

        crop_top = int(height_from_center - height_from_center / self.factor)
        crop_left = int(width_from_center - width_from_center / self.factor)

        crop_width = int((width_from_center / self.factor) * 2)
        crop_height = int((height_from_center / self.factor) * 2)

        return functional.resized_crop(
            img=x,
            top=crop_top,
            left=crop_left,
            height=crop_height,
            width=crop_width,
            size=[x.shape[-2], x.shape[-1]],
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )


class RandomDiscreteRotationElement(torchvision.transforms.RandomRotation):

    def __init__(
            self, degrees, num_group_elements
    ):
        super(RandomDiscreteRotationElement, self).__init__(degrees=degrees, fill=0)

        self.elements = torch.linspace(float(self.degrees[0]), float(self.degrees[1]), num_group_elements)

    @staticmethod
    def get_params(elements: torch.Tensor) -> float:
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        return float(elements[torch.randint(high=elements.shape[0], size=[1]).item()])

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.
        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * functional._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.elements)

        return functional.rotate(img, angle, self.resample, self.expand, self.center, fill)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')