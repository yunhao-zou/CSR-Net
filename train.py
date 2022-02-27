import argparse
import auxil
import os
from dataset import *

import torch
import torch.nn.parallel
from torchvision.transforms import *
from torch.nn.functional import l1_loss, mse_loss
from torch.nn.modules.loss import _Loss

import models.model_csrnet as csrnet
import numpy as np
from auxil import str2bool
from torch.utils.data.dataset import random_split

import matplotlib.pyplot as plt

def load_hyper(args):
    if args.dataset == 'BITflower':
        data_path = '/data/hsi_classification_test/flower_full.h5'
        # f = h5py.File(data_path, 'r')
        full_hyper = BITDataset(data_path)
        numberofclass = 60
        data_shape = None
        n_train = int(len(full_hyper)*args.tr_percent)
        n_test = len(full_hyper) - n_train
        train_hyper, test_hyper = random_split(full_hyper, [n_train, n_test])
        patchesLabels = None
        bands = 256
    
    else:
        data, label, numclass = auxil.loadData(args.dataset, num_components=args.components)
        data_shape = data.shape
        patchesLabels, pixels, labels = auxil.createImageCubes(data, label, windowSize=args.spatialsize, removeZeroLabels = True)
        # print(pixels.shape)
        bands = pixels.shape[-1]; numberofclass = len(np.unique(labels))
        x_train, x_test, y_train, y_test = auxil.split_data(pixels, labels, args.tr_percent)
        # del pixels, labels
        train_hyper = Dataset((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"),y_train), None)
        test_hyper  = Dataset((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"),y_test), None)
        full_hyper = Dataset((np.transpose(pixels, (0, 3, 1, 2)).astype("float32"),labels), None)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    full_loader  = torch.utils.data.DataLoader(full_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    return patchesLabels, full_loader, train_loader, test_loader, numberofclass, bands, data_shape


def train(trainloader, model, criterion, smooth_criterion, optimizer, epoch, use_cuda, args):
    model.train()
    accs   = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print('epoch:'+str(epoch)+'  |  progress: '+str(batch_idx)+'/'+str(len(trainloader)))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda().long()
        outputs = model(inputs)
        loss1 = criterion(outputs, targets)
        # print(loss1)
        losses[batch_idx] = loss1.item()
        loss2 = smooth_criterion(model)
        # print('loss1:', loss1)
        # print('loss2:', loss2)
        loss = loss1 + args.mu*loss2
        accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # WeightClipper(model)
        # print('next iteration!')
    return (np.average(losses), np.average(accs))


def test(testloader, model, criterion, epoch, use_cuda):
    model.eval()
    accs   = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda().long()
            outputs = model(inputs)
            losses[batch_idx] = criterion(outputs, targets).item()
            accs[batch_idx] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return (np.average(losses), np.average(accs))




def predict(testloader, model, criterion, use_cuda):
    model.eval()
    predicted = []
    with torch.no_grad():	
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda: inputs = inputs.cuda()
            [predicted.append(a) for a in model(inputs).data.cpu().numpy()] 
    return np.array(predicted)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def WeightClipper(m):
    # for p in m.parameters():
    m.filter.weight.data.clamp_(min=0)
    # for p in m.filter_set:
    #     p.weight.data.clamp_(min=0)
        # p.weight.data.div_(torch.max(p.weight.data))


class SmoothLoss(_Loss):
    """
    Add Smooth Constraints
    """
    def __init__(self, reduction='sum'):
        super(SmoothLoss, self).__init__(reduction=reduction)

    def forward(self, model):
        loss = 0
        # for m in model.filter_set:
        #     w = torch.squeeze(m.weight)
        #     print(w.shape)
        #     loss += mse_loss(w[:-1], w[1:], reduction='sum')
        m = model.filter
        w = torch.squeeze(m.weight.data)
        loss += mse_loss(w[:, :-1], w[:, 1:], reduction='sum')
        return loss

def main():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    parser.add_argument('--dataset', default='IP', type=str, help='dataset (options: IP, PU, SV, KSC)')
    parser.add_argument('--tr_percent', default=0.15, type=float, help='samples of train set')
    parser.add_argument('--tr_bsize', default=100, type=int, help='mini-batch train size (default: 100)')
    parser.add_argument('--te_bsize', default=1000, type=int, help='mini-batch test size (default: 1000)')
    parser.add_argument('--depth', default=32, type=int, help='depth of the network (default: 32)')
    parser.add_argument('--alpha', default=48, type=int, help='number of new channel increases per depth (default: 12)')
    parser.add_argument('--inplanes', dest='inplanes', default=16, type=int, help='bands before blocks')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='to use basicblock (default: bottleneck)')
    parser.add_argument('--spatialsize', dest='spatialsize', default=11, type=int, help='spatial-spectral patch dimension')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--crf', type=str2bool)
    parser.add_argument('--crf_channel', type=int)
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--resume', type=str2bool, default='false')

    parser.set_defaults(bottleneck=True)
    
    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    patchesLabels, full_loader, train_loader, test_loader, num_classes, n_bands, data_shape = load_hyper(args)
    print('[i] Dataset finished!')
    # Use CUDA
    use_cuda = torch.cuda.is_available()
    if use_cuda: torch.backends.cudnn.benchmark = True

    if args.spatialsize < 9: avgpoosize = 1
    elif args.spatialsize <= 11: avgpoosize = 2
    elif args.spatialsize == 15: avgpoosize = 3
    elif args.spatialsize == 19: avgpoosize = 4
    elif args.spatialsize == 21: avgpoosize = 5
    elif args.spatialsize == 27: avgpoosize = 6
    elif args.spatialsize == 29: avgpoosize = 7
    elif args.spatialsize == 64: avgpoosize = 15
    else: print("[i] spatialsize not supported")
    psize = args.spatialsize // avgpoosize
    # model = HOUR.hourglass(n_bands, num_classes, avgpoosize)
    model = csrnet.CSRNet(args.crf, args.crf_channel, args.depth, args.alpha, num_classes, n_bands, avgpoosize, args.inplanes, psize, bottleneck=args.bottleneck) # for PyramidNet
    if use_cuda: model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    smooth_criterion = SmoothLoss()
    #optimizer = torch.optim.Adam(model.parameters())
    paras = dict(model.named_parameters())
    paras_group = []
    for k, v in paras.items():
        if 'filter' in k:
            paras_group += [{'params': [v], 'weight_decay': args.weight_decay}]
        else:
            paras_group += [{'params': [v], 'weight_decay': 1e-4}]
    # optimizer = torch.optim.Adam(paras_group, args.lr)
    optimizer = torch.optim.SGD(paras_group, args.lr,
                                momentum=args.momentum,nesterov=True)

    best_acc = -1
    init_epoch = 0
    if args.resume:
        # checkpoint = torch.load('current_model/' + args.dataset + '_'+ str(args.crf_channel) + '_w' + str(args.weight_decay) + '_mu' + str(args.mu) + "_patten2.pth")
        checkpoint = torch.load('current_model/' + args.dataset + '_CSRNet.pth')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    for epoch in range(init_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_loss, train_acc = train(train_loader, model, criterion, smooth_criterion, optimizer, epoch, use_cuda, args)
        with torch.no_grad():
            test_loss, test_acc = test(test_loader, model, criterion, epoch, use_cuda)

        print("EPOCH", epoch, "Train Loss", train_loss, "Train Accuracy", train_acc, end=', ')
        print("Test Loss", test_loss, "Test Accuracy", test_acc)
        # save model
        torch.save(state, 'current_model/' + args.dataset + '_CSRNet.pth')
        if test_acc > best_acc:
            state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, 'best_model/' + args.dataset + '_CSRNet.pth')
            best_acc = test_acc

    checkpoint = torch.load('best_model/' + args.dataset + '_CSRNet.pth')
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    with torch.no_grad():
        test_loss, test_acc = test(test_loader, model, criterion, epoch, use_cuda)
    print("FINAL:      LOSS", test_loss, "ACCURACY", test_acc)
    prediction = np.argmax(predict(full_loader, model, criterion, use_cuda), axis=1)
    de_map = np.zeros(patchesLabels.shape, dtype=np.int32)
    index = 0
    for i in range(patchesLabels.shape[0]):
        if patchesLabels[i] == 0:
            de_map[i] = 0
        else:
            de_map[i] = prediction[index]
            index = index + 1
    de_map = np.reshape(de_map, (data_shape[0], data_shape[1]))
    w, h = de_map.shape
    plt.figure(figsize=[h/100.0, w/100.0])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    plt.axis('off')
    plt.imshow(de_map, cmap='jet')
    plt.savefig(os.path.join('plot/CSRNet_' + args.dataset + '.png'), format='png')
    plt.close()

    classification, confusion, results = auxil.reports(np.argmax(predict(test_loader, model, criterion, use_cuda), axis=1), np.array(test_loader.dataset.__labels__()), args.dataset)
    print(args.dataset, results)
    # import ipdb; ipdb.set_trace()
    str_res = np.array2string(np.array(results), max_line_width=200)
    print(str_res)
    log = ('Dataset = %s, patch size = %d, Loss = %.8f, Accuracy = %4.4f\nResults = %s\n') % (args.dataset, args.spatialsize, test_loss, test_acc, str_res)
    with open(os.path.join('results', 'csrnet_result.txt'), 'a') as f:
        f.write(log)
    # ---------------save csr---------------
    # csr = []
    # for fil in model.filter_set:
    #     # print(f.weight.data)
    #     fil = fil.cpu()
    #     csr.append(fil.weight.data.numpy())
    # csr = model.filter.weight.data.cpu()
    # csr = np.array(csr, dtype='float32')
    # csr = csr.reshape(args.crf_channel, n_bands)
    # scipy.io.savemat('csr_results/' + args.dataset + str(args.crf_channel) + '_w' + str(args.weight_decay) + '_mu' + str(args.mu) + "_csr.mat", {'csr': csr})

if __name__ == '__main__':
    torch.set_num_threads(1)
    main()

