# MultiAug

MultiAug is a multi-modal data augmentation library for use in machine learning. The library aims to provide the following functionality:

* For datasets where there are multiple modalities describing the same sample point (i.e. tabular data and image data), generate new data points by augmenting corresponding samples in the different modalities
* Augmentation for 3D images
* Augmentation for tabular data

Functionally, the library presents a similar API to [imgaug](https://github.com/aleju/imgaug) python library

## Install

```
pip install MultiAug
```

## Current Features

3D image augmentation

* Random rotation

Tabular data augmentation

* Featurewise Gaussian noise

## API

Operators:

* The `OneOf()` method with apply one of the transformations provided in the list to the corresponding modality

    * `augment` can either be a fraction of the dataset to augment or a predetermined list of indices in the dataset that you want to augment
    * `image3d_transforms` list of possible augmentations to apply to 3D images
    * `tabular_transforms` list of possible augmentations to apply to tabular data

## Examples

Randomly augment 50% of the data by rotating 3D images about the x, y, z axes by `angle` degrees

```python
import multiaug.augmenters as aug
a = aug.OneOf(augment=0.5, image3d_transforms=[aug.image3d_augmenters.Rotate3d(angle=5)])
data, labels = load_data() # must return (B x H x W x D, [int]) where [int] is categorical integers
new_data, new_labels = a.apply_image3d(data, labels)
```

Randomly augment 50% of the data by applying featurewise Gaussian noise as 10% of the variance of each feature


```python
import multiaug.augmenters as aug
a = aug.OneOf(augment=0.5, tabular_transforms=[aug.tabular_augmenters.GaussianPerturbation(method='variance', fraction=0.1)])
data, labels = load_data() # must return (B x Feats, [int]) where [int] is categorical integers
new_data, new_labels = a.apply_tabular(data, labels)
```

Randomly augment 50% of the data by applying rotation to 3D images and featurewise Guassian noise to the corresponding tabular data

```python
import multiaug.augmenters as aug
a = aug.OneOf(augment=0.5, image3d_transforms=[aug.image3d_augmenters.Rotate3d(angle=5)], tabular_transforms=[aug.tabular_augmenters.GaussianPerturbation(method='variance', fraction=0.1)])
image_data, labels = load_data() # must return (B x H x W x D, [int]) where [int] is categorical integers
tabular_data, _ = load_data() # must return (B x Feats, [int]) where [int] is categorical integers
new_image_data, new_labels = a.apply_image3d(image_data, labels)
new_tabular_data, _ = a.apply_tabular(tabular_data, labels)
```
