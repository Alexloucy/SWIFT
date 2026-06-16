from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    RandomErasing,
    ColorJitter,
    ToTensor,
    RandomGrayscale
)

INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


def build_transform(cfg, is_train=True):
    if is_train:
        return _build_transform_train(cfg, cfg.INPUT.TRANSFORMS)
    else:
        return _build_transform_test(cfg, cfg.INPUT.TRANSFORMS)


def _build_transform_train(cfg, transform_choices):
    transform_train = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    transform_train += [Resize(max(cfg.INPUT.SIZE), interpolation=interp_mode)]

    if "random_resized_crop" in transform_choices:
        transform_train += [
            RandomResizedCrop(
                size=cfg.INPUT.SIZE,
                scale=cfg.INPUT.RRCROP_SCALE,
                interpolation=interp_mode,
            )
        ]
    else:
        transform_train += [CenterCrop(cfg.INPUT.SIZE)]

    if "random_flip" in transform_choices:
        transform_train += [RandomHorizontalFlip()]

    if "color_jitter" in transform_choices:
        transform_train += [
            ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.0,
            )
        ]
    
    if "gray_scale" in transform_choices:
        transform_train += [RandomGrayscale(p=0.3)]

    # Convert to tensor before any tensor-only transforms
    transform_train += [ToTensor()]

    if "normalize" in transform_choices:
        transform_train += [
            Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ]

    # RandomErasing expects a tensor image; apply after ToTensor/Normalize
    if "random_erase" in transform_choices:
        transform_train += [
            RandomErasing(p=0.7, scale=(0.10, 0.40))
        ]

    transform_train = Compose(transform_train)
    print("Transform for Train: {}".format(transform_train))

    return transform_train


def _build_transform_test(cfg, transform_choices):
    transform_test = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]

    transform_test += [Resize(max(cfg.INPUT.SIZE), interpolation=interp_mode)]

    transform_test += [CenterCrop(cfg.INPUT.SIZE)]

    transform_test += [ToTensor()]

    if "normalize" in transform_choices:
        transform_test += [
            Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ]

    transform_test = Compose(transform_test)
    print("Transform for Test: {}".format(transform_test))

    return transform_test
