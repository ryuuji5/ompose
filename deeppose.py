from chainer import FunctionSet, Variable
import chainer.functions as F

class Deeppose(FunctionSet):
    """Single-GPU AlexNet with Normalization layers replaced by
    BatchNormalization."""

    insize = 220

    def __init__(self):
        super(Deeppose, self).__init__(
            conv1 = F.Convolution2D(  3,  96, 11, stride=4),
            bn1   = F.BatchNormalization( 96),
            conv2 = F.Convolution2D( 96, 256,  5, pad=2),
            bn2   = F.BatchNormalization(256),
            conv3 = F.Convolution2D(256, 384,  3, pad=1),
            conv4 = F.Convolution2D(384, 384,  3, pad=1),
            conv5 = F.Convolution2D(384, 256,  3, pad=1),
            fc6   = F.Linear(9216, 4096),
            fc7   = F.Linear(4096, 4096),
            fc8   = F.Linear(4096, 28),
        )

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return F.sigmoid(h, t), F.accuracy(h, t)
