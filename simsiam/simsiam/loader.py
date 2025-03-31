# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import ImageFilter
import random


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, sec_transform = None):
        self.base_transform = base_transform
        self.sec_transform = sec_transform

    def __call__(self, x):
        q = self.base_transform(x)
        if self.sec_transform is not None:
            k = self.sec_transform(x)
        else:
            k = self.base_transform(x)
        return [q, k]


class MultiCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, sec_transform = None):
        self.base_transform = base_transform
        self.sec_transform = sec_transform

    def __call__(self, x):
        q = self.base_transform(x)
        if self.sec_transform is not None:
            k1, k2 = self.sec_transform(x), self.sec_transform(x)
        else:
            k1, k2 = self.base_transform(x), self.base_transform(x)
        return [q, k1, k2]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
