import chainer
import chainer.links as L
import chainer.functions as F


class GGCNN(chainer.Chain):

    def __init__(self):
        super(GGCNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 32, 9, 3, 3)  # (32, 100, 100)
            self.conv2 = L.Convolution2D(32, 16, 5, 2, 2)  # (16, 50, 50)
            self.conv3 = L.Convolution2D(16, 8, 3, 2, 1)  # (8, 25, 25)

            self.deconv1 = L.Deconvolution2D(8, 8, 3, 2, pad=1,
                                             outsize=(50, 50))
            self.deconv2 = L.Deconvolution2D(8, 16, 5, 2, pad=2,
                                             outsize=(100, 100))
            self.deconv3 = L.Deconvolution2D(16, 32, 9, 3, pad=3,
                                             outsize=(300, 300))

            self.conv_pos = L.Convolution2D(32, 1, 2, dilate=0)

            self.conv_sin = L.Convolution2D(32, 1, 2, dilate=0)

            self.conv_cos = L.Convolution2D(32, 1, 2, dilate=0)

            self.conv_width = L.Convolution2D(32, 1, 2, dilate=0)

    def forward(self, x):
        """

        Args:

        """
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        encoded = F.relu(self.conv3(h))

        h = F.relu(self.deconv1(encoded))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))

        pos = self.conv_pos(h)
        sin = self.conv_sin(h)
        cos = self.conv_cos(h)
        width = self.conv_width(h)

        return pos, sin, cos, width


if __name__ == '__main__':
    import numpy as np

    img = np.ones((1, 1, 300, 300), 'f')
    model = GGCNN()
    pos, sin, cos, width = model(img)
    print(pos.shape, sin.shape, cos.shape, width.shape)
