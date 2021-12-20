import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--model_name", type=str, default='')
parser.add_argument("--seed", type=int, default=45)

parser.add_argument("--batch_size", type=int, default=160)
parser.add_argument("--max_epoch", type=int, default=150)

parser.add_argument("--pretrain_lr", type=float, default=0.5)
parser.add_argument("--pretrain_epoch", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--lr_decay", action='store_true')
parser.add_argument("--lr_decay_epoch", type=int, default=50)
parser.add_argument("--lr_decay_rate", type=float, default=0.1)
parser.add_argument("--lr_min", type=float, default=0.001)

parser.add_argument("--weight_decay", type=float, default=1e-5)

parser.add_argument("--opt", type=str, default='sgd')
parser.add_argument("--bag_size", type=int, default=0)
parser.add_argument("--loss_weight", action='store_true')

parser.add_argument("--mil", type=str, default='att')

parser.add_argument("--vat_threshold", type=float, default=0.1)
parser.add_argument("--vat_alpha", type=float, default=0.1)
parser.add_argument("--vat_iter", type=int, default=1)
parser.add_argument("--vat_xi", type=float, default=1e-6)
parser.add_argument("--vat_eps", type=float, default=2.5)

parser.add_argument("--at_alpha", type=float, default=1.0)
parser.add_argument("--at_eps", type=float, default=0.05)

# args = parser.parse_args(sys.argv)
args, unknown = parser.parse_known_args()
