from DataAugmentation import DataAugmentation

aug_params = {
	'data_root': './SUNRGBD/',
    'img_dimension': 128,
    'max_rotate_angle': 10,
    'rotate_zero_padding': False,
    'random_flip': True,
    'saturation_range': (0.5, 2),
    'contrast_range': (0.5, 2),
    'brightness_range': (0.5, 2),
    'iteration': 20
}

data = DataAugmentation(**aug_params)
data.augmente()