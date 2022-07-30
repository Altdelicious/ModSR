from train import train, test
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--GT-path', type=str, default='/home/bhavesh/data/291')
parser.add_argument('--test-GT-path', type=str, default='/home/bhavesh/data/Set5/HR')
parser.add_argument('--parameter-save-path', type=str, default='parameters/x2')
parser.add_argument('--parameter-restore-path', type=str, default=None)
parser.add_argument('--parameter-name', type=str, default='vdsr_x2.pth')
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--learning-rate', type=float, default=1e-1)
parser.add_argument('--train-iteration', type=int, default=50000)
parser.add_argument('--log-step', type=int, default=50)
parser.add_argument('--validation-step', type=int, default=200)
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--patch-size', type=int, default=64)
parser.add_argument('--lazy-load', action='store_true', default=False)
parser.add_argument('--rgb', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--gpu', type=str, default='1')
args = parser.parse_args()
print('[*] Using GPU: {}'.format(args.gpu), flush=True)


test(args) if(args.test) else train(args)