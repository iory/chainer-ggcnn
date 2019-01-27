import chainer
import chainer.functions as F


class GGCNNTrainChain(chainer.Chain):

    def __init__(self, predictor):
        super(GGCNNTrainChain, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def predict(self, imgs):
        pred_pos, pred_sin, pred_cos, pred_width = self.predictor(imgs)
        return pred_pos, pred_sin, pred_cos, pred_width

    def forward(self,
                imgs,
                positions,
                sines,
                cosines,
                widthes):
        pred_pos, pred_sin, pred_cos, pred_width = self.predictor(imgs)

        pos_loss = F.mean_squared_error(positions, pred_pos)
        sin_loss = F.mean_squared_error(sines, pred_sin)
        cos_loss = F.mean_squared_error(cosines, pred_cos)
        width_loss = F.mean_squared_error(widthes, pred_width)
        loss = pos_loss + sin_loss + cos_loss + width_loss
        chainer.reporter.report(
            {'loss': loss,
             'pos_loss': pos_loss,
             'sin_loss': sin_loss,
             'cos_loss': cos_loss,
             'width_loss': width_loss}, self)
        return loss
