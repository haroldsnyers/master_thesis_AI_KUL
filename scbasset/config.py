from xmlrpc.client import Boolean
import attr
import configargparse

@attr.s
class Config:
    parser_args = attr.ib(default=None)
    out_dir = attr.ib(default='output', type=str)
    bottleneck = attr.ib(default=32, type=int)
    batch_size = attr.ib(default=128, type=int)
    learning_rate = attr.ib(default=0.01, type=float)
    epochs = attr.ib(default=1000, type=int)
    h5_file = attr.ib(default=None, type=str)
    model_name = attr.ib(default='scbasset', type=str)
    residual_model = attr.ib(default=False, type=Boolean)
    activation_fct = attr.ib(default='gelu', type=str)
    seq_length = attr.ib(default=1344, type=int)
    num_heads = attr.ib(default=8, type=int)
    repeat = attr.ib(default=4, type=int)
    num_transforms = attr.ib(default=6, type=int)
    cuda = attr.ib(default=2, type=int)
    logs = attr.ib(default='logs', type=str)
    weights = attr.ib(default=None, type=str)
    tower_multiplier = attr.ib(default=1.122, type=float)
    version = attr.ib(default=None, type=str)

    def load_args(self):
        self.out_dir = self.parser_args.out_path
        self.bottleneck_size = self.parser_args.bottleneck
        self.batch_size = self.parser_args.batch_size
        self.learning_rate = self.parser_args.lr
        self.epochs = self.parser_args.epochs
        self.h5_file = self.parser_args.h5
        self.model_name = self.parser_args.model
        self.residual_model = self.parser_args.feature
        self.activation_fct = self.parser_args.activation
        self.seq_length = self.parser_args.seq_length
        self.num_heads = self.parser_args.num_heads
        self.repeat = self.parser_args.repeat
        self.num_transforms = self.parser_args.num_transforms
        self.cuda = self.parser_args.cuda
        self.weights = self.parser_args.weights
        self.tower_multiplier = self.parser_args.tower_multiplier
        self.version = self.parser_args.version

    def make_parser(self):
        parser = configargparse.ArgParser(
            description="train scBasset on scATAC data")
        parser.add_argument('--h5', type=str,
                            help='path to h5 file.')
        parser.add_argument('--bottleneck', type=int, default=32,
                            help='size of bottleneck layer. Default to 32')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='batch size. Default to 128')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='learning rate. Default to 0.01')
        parser.add_argument('--epochs', type=int, default=1000,
                            help='Number of epochs to train. Default to 1000.')
        parser.add_argument('--out_path', type=str, default='output',
                            help='Output path. Default to ./output/')
        parser.add_argument('--logs', type=str, default='logs',
                            help='Logs path. Default to ./logs/')
        parser.add_argument('--cuda', type=int, default=2,
                            help='CUDA device number, Default to 2')
        parser.add_argument('--model', type=str, default='scbasset',
                            help='Which model to choose from, default to scbasset. Choice between scbasset and tfBanformer')
        parser.add_argument('--residual', dest='feature', action='store_true',
                            help='residual flag, if set then residual tower, if --no-residual or nothing, then no residual layers')
        parser.add_argument('--no-residual', dest='feature', action='store_false')
        parser.set_defaults(feature=False)
        parser.add_argument('--activation', type=str, default='gelu', help='activation function name, default to gelu, other relu')
        parser.add_argument('--seq_length', type=int, default=1344, help='sequence length')
        parser.add_argument('--num_heads', type=int, default=8, help='number of heads for transformer block')
        parser.add_argument('--repeat', type=int, default=6, help='Number of conv blocks in conv tower')
        parser.add_argument('--num_transforms', type=int, default=4, help='Number of transformer blocks in transformer tower')
        parser.add_argument('--weights', type=str, default=None, help='path to base model weight')
        parser.add_argument('--tower_multiplier', type=float, default=1.122, help='tower multiplier, default 1.122')
        parser.add_argument('--version', type=str, default=None, help='training version')

        self.parser_args = parser.parse_args()