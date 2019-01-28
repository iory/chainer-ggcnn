import chainer
import chainer.links as L
import chainer.functions as F


class GGCNN(chainer.Chain):

    def __init__(self, use_bn=False):
        super(GGCNN, self).__init__()
        self.use_bn = use_bn
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

            self.conv_pos = L.Convolution2D(32, 1, 2, pad=1)

            self.conv_sin = L.Convolution2D(32, 1, 2, pad=1)

            self.conv_cos = L.Convolution2D(32, 1, 2, pad=1)

            self.conv_width = L.Convolution2D(32, 1, 2, pad=1)

            if use_bn:
                self.bn1 = L.BatchNormalization(32)
                self.bn2 = L.BatchNormalization(16)
                self.bn3 = L.BatchNormalization(8)

                self.dbn1 = L.BatchNormalization(8)
                self.dbn2 = L.BatchNormalization(16)
                self.dbn3 = L.BatchNormalization(32)

    def predict(self, imgs):
        pred_poses = []
        pred_angles = []
        pred_widthes = []
        for img in imgs:
            C, H, W = img.shape
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                x = chainer.Variable(self.xp.asarray(img[self.xp.newaxis]))
                pred_pos, pred_sin, pred_cos, pred_width = self.__call__(x)
            pred_pos = pred_pos[0].data
            pred_sin = pred_sin[0].data
            pred_cos = pred_cos[0].data
            pred_width = pred_width[0].data
            pred_angle = self.xp.arctan2(pred_sin, pred_cos) / 2.0
            pred_pos = chainer.backends.cuda.to_cpu(pred_pos)
            pred_angle = chainer.backends.cuda.to_cpu(pred_angle)
            pred_width = chainer.backends.cuda.to_cpu(pred_width) * 150.0
            pred_poses.append(pred_pos)
            pred_angles.append(pred_angle)
            pred_widthes.append(pred_width)
        return pred_poses, pred_angles, pred_widthes

    def forward(self, x):
        """

        Args:

        """
        if self.use_bn:
            h = F.relu(self.bn1(self.conv1(x)))
            h = F.relu(self.bn2(self.conv2(h)))
            encoded = F.relu(self.bn3(self.conv3(h)))

            h = F.relu(self.dbn1(self.deconv1(encoded)))
            h = F.relu(self.dbn2(self.deconv2(h)))
            h = F.relu(self.dbn3(self.deconv3(h)))

            pos = self.conv_pos(h)[:, :, :300, :300]
            sin = self.conv_sin(h)[:, :, :300, :300]
            cos = self.conv_cos(h)[:, :, :300, :300]
            width = self.conv_width(h)[:, :, :300, :300]
        else:
            h = F.relu(self.conv1(x))
            h = F.relu(self.conv2(h))
            encoded = F.relu(self.conv3(h))

            h = F.relu(self.deconv1(encoded))
            h = F.relu(self.deconv2(h))
            h = F.relu(self.deconv3(h))

            pos = self.conv_pos(h)[:, :, :300, :300]
            sin = self.conv_sin(h)[:, :, :300, :300]
            cos = self.conv_cos(h)[:, :, :300, :300]
            width = self.conv_width(h)[:, :, :300, :300]

        return pos, sin, cos, width


if __name__ == '__main__':
    import numpy as np

    img = np.ones((4, 1, 300, 300), 'f')
    model = GGCNN()
    pos, sin, cos, width = model(img)
    print(pos.shape, sin.shape, cos.shape, width.shape)
