import argparse
import torch
import models.presnet_atten as PYRM
import numpy as np
import scipy.io
parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
parser.add_argument('--dataset', default='IP', type=str, help='dataset (options: IP, PU, SV, KSC)')
parser.add_argument('--crf_channel', type=int, default=10)
parser.add_argument('--mu', type=float, default=0.1)
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
args = parser.parse_args()
model = PYRM.pResNet(True, 10, 32, 48, 16, 200, 2, 16, bottleneck=True)
checkpoint = torch.load('best_model/' + args.dataset + str(args.crf_channel) + '_w' + str(args.weight_decay) + '_mu' + str(args.mu) + ".pth")
model.load_state_dict(checkpoint['state_dict'])
# csr = []
# for f in model.filter_set:
#     print(f.weight.data)
#     csr.append(f.weight.data.numpy())
# csr = np.array(csr, dtype='float32')
# csr = csr.reshape(3, 200)
# scipy.io.savemat('csr_results/' + args.dataset + str(args.crf_channel) + '_w' + str(args.weight_decay) + '_mu' + str(args.mu) + "_csr.mat", {'csr': csr})
csr = model.filter.weight.data.cpu()
csr = np.array(csr, dtype='float32')
csr = csr.reshape(args.crf_channel, 200)
scipy.io.savemat('csr_results/' + args.dataset + str(args.crf_channel) + '_w' + str(args.weight_decay) + '_mu' + str(args.mu) + "_csr2.mat", {'csr': csr})