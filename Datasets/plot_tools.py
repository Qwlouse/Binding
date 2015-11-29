#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt
import seaborn as sns


def plot_groups(groups, ax):
    mask = (groups == 0)
    sns.heatmap(groups, mask=mask, square=True, cmap='viridis_r',
                xticklabels=False, yticklabels=False, cbar=False, ax=ax)


def plot_input_image(img, ax):
    mask = (img == 0)
    sns.heatmap(img, mask=mask, square=True, xticklabels=False,
                yticklabels=False, cmap='Greys', cbar=False, ax=ax)
