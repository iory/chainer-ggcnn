import pkg_resources


__version__ = pkg_resources.get_distribution(
    'chainer_ggcnn').version


import chainer_ggcnn.links  # NOQA
import chainer_ggcnn.utils  # NOQA
