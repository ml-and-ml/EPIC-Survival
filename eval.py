import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='EPIC-Survitest')
parser.add_argument('--lr', default=0.1, type=float, metavar='float', help='learning rate')
parser.add_argument('--dropout', default=.5, type=float, metavar='0 to 1', help='dropout rate for fc layers')
parser.add_argument('--out', default=1, type=int, metavar='0 or 1', help='0 for testing, 1 to save experiments')
parser.add_argument('-n', '--n-cluster', default=10, type=int, metavar='int', help='# of clusters')
parser.add_argument('-w', '--waist', default=64, type=int, metavar='int', help='encoder length')
parser.add_argument('--development', default=0, type=int, metavar='0 or 1', help='for quick testing')
parser.add_argument('-e', '--epochs', default=3000, type=int, metavar='int', help='# training epochs')
parser.add_argument('--train_full', default=0, type=int, metavar='0 or 1', help='0 for testing on test')
parser.add_argument('--p-batch-size', default=32, type=int, metavar='int', help='part loader batch size')
parser.add_argument('--t-batch-size', default=1024, type=int, metavar='int', help='tile loader batch size')
parser.add_argument('-s', '--sample', default=100, type=int, metavar='int', help='# tiles sampled per slide')
parser.add_argument('--verbose', default=0, type=int, metavar='0 or 1', help='set to 1 to print extra training details')
parser.add_argument('-k', '--topk', default=3, type=int, metavar='int', help='# top tiles to sample from for part')
parser.add_argument('--subsample', default=1, type=float, metavar='<1', help='percent slides subsampled per epoch')
parser.add_argument('--clusteringlambda', default=1, type=float, metavar='0 to 1', help='constraint weight in loss')
parser.add_argument('--weightdecay', default=1e-4, type=float, metavar='0 to 1', help='weight decat')
parser.add_argument('--level', default=0, type=int, metavar='0 or 1', help='tile resolution')
parser.add_argument('--slide_path', default='./', type=str, metavar='dir', help='path to svs directory')
parser.add_argument('--library', default='./', type=str, metavar='file', help='path to tile library csv')
parser.add_argument('--checkpoint', default='./', type=str, metavar='file', help='path model checkpoint')

import os
import utils
import torch
import random
import dataloaders
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
from models import resblock18,SurvivalResNet
from lifelines.utils import concordance_index


def main():

    print('!! -------- Initializing EPIC-Survitest -------- !!')
    global args, gpu
    args = parser.parse_args()
    gpu = torch.device('cuda')
    root_output = '#SET ROOT OUTPUT DIRECTORY#'.format(args.dataset)

    save_path = '{}lr_{}d_{}n_{}p_{}w_{}b_{}sl_{}lam_{}strat_0.0001wd_default_large'.format(args.lr,
                                                            args.dropout,
                                                            args.n_cluster,
                                                            args.topk,
                                                            args.waist,
                                                            args.p_batch_size,
                                                            args.decay_intertest,
                                                            args.clusteringlambda,
                                                            args.stratify)

    out_dir = os.path.join(root_output, save_path)


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'part_viz')):
        os.mkdir(os.path.join(out_dir, 'part_viz'))

    ###################################################################################################################

    print('Initialization [1/4] ........ Building Model Architecture')


    feature_extractor = nn.DataParallel(resblock18(args.waist)).to(gpu)
    model = SurvitestResNet(feature_extractor, args.waist, num_clusters=args.n_cluster, dropout=args.dropout).to(gpu)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer'])

    data = pd.read_csv(args.library)
    test_library = data[data.Split == 'test'].reset_index(drop=True)

    normalize = tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    augmentation = tf.Compose([tf.CenterCrop(224), tf.ToTensor(), normalize])

    tile_dataset_test = dataloaders.TileLoader(args, test_library,augmentation)
    tile_loader_test = torch.utils.data.DataLoader(tile_dataset_test,
                                                   batch_size=args.t_batch_size,
                                                   shuffle=False,
                                                   num_workers=10)

    global_centroids = checkpoint['g_centers']
    local_centroids = checkpoint['l_centers']
    top_tiles = checkpoint['top_tiles']
    part_information = checkpoint['part_information']
    assignments = checkpoint['assignments']

    supervised_loss = testLoss().to(gpu)


    ###################################################################################################################


    test_set = test_library.SlideID.unique()
    test_embedding = torch.randn(len(test_set) * 1000, args.waist)
    test_ids = torch.randint(0, args.n_cluster, (len(test_set) * 1000,))
    test_indices = torch.randint(0, args.n_cluster, (len(test_set) * 1000,))
    test_assignments = torch.randint(0, args.n_cluster, (len(test_set) * 1000,))
    test_subset_library = test_library.groupby('SlideID').apply(lambda x: x.sample(n=1000, replace=False))
    #n = number of tiles pulled from each slide. might need to decrease in with slide has less issue
    test_subset_library = test_subset_library.droplevel(0).reset_index(drop=True)
    tile_loader_test.dataset.assignment(test_subset_library)
    locations = torch.LongTensor(len(test_subset_library), 3)
    model.eval()
    with torch.no_grad():  # pass subset tiles through base model to retrieve embeddings
        for i, (input, _, id, local_ind, index) in enumerate(tile_loader_test):
            output = feature_extractor(input.to(gpu))
            test_ids[(i * args.t_batch_size):(i * args.t_batch_size + len(input))] = id
            test_indices[(i * args.t_batch_size):(i * args.t_batch_size + len(input))] = local_ind
            emb = output.detach()
            test_embedding[(i * args.t_batch_size):(i * args.t_batch_size + len(input))] = emb
            batch_assignments = assign(emb, global_centroids)
            test_assignments[(i * args.t_batch_size):(i * args.t_batch_size + len(input))] = batch_assignments
            batch_data = test_subset_library.iloc[local_ind]
            slide = torch.from_numpy(batch_data.SlideID.values)
            x = torch.from_numpy(batch_data.x.values.astype(int))
            y = torch.from_numpy(batch_data.y.values.astype(int))
            locations[index] = torch.stack([slide, x, y], 1)

    for slide in test_ids.unique():
        slide_embedding = test_embedding[(test_ids == slide).nonzero().squeeze()]
        slide_assignments = test_assignments[(test_ids == slide).nonzero().squeeze()]
        slide_indices = test_indices[(test_ids == slide).nonzero().squeeze()]
        if slide.item() in local_centroids:
            slide_centroids = local_centroids[slide.item()]
        else:
            slide_centroids = torch.randn(args.n_cluster, args.waist)
        slide_centroids = calculate_centroids(slide_embedding, slide_assignments, slide_centroids)
        local_centroids[slide.item()] = slide_centroids
        part_indices, part_weights, c_mask = part_selection(slide_embedding, slide_centroids, slide_assignments)
        top_tiles[slide.item()] = slide_indices[part_indices]
        assignments[slide.item()] = slide_assignments[part_indices]
        part_information[slide.item()] = list(zip(part_weights, c_mask))



    part_dataset_test = dataloaders.PartLoader(args, test_subset_library,top_tiles, augmentation, part_information)
    part_loader_test = torch.utils.data.DataLoader(part_dataset_test,
                                                   batch_size=args.p_batch_size,
                                                   shuffle=False,
                                                   num_workers=10,
                                                   pin_memory=True)


    model.eval()
    #gradient is measured to visualize attribution
    for i, (parts, duration, event, weights, mask, val_id, index) in enumerate(
            part_loader_test):
        img = [x.to(gpu) for x in parts]
        risk, hx, _ = model(img, mask, weights, True)
        ci = concordance_index(duration.numpy(), risk.detach().cpu().numpy(), event.numpy())

        print('CI: {}'.format(ci))
        hx.retain_grad()
        loss = supervised_loss(risk, duration, event)
        loss.backward()

        hx_grad = hx.grad.detach().cpu()
        attribution_full = torch.abs(hx_grad.reshape(-1, args.n_cluster, args.waist).mean(-1))




    ordered_risks, order_idx = torch.sort(risk.squeeze())
    ordered_ids = val_id[order_idx]
    risks_to_display = ordered_ids.long()
    ordered_attr = attribution_full[order_idx]
    attr_to_display = ordered_attr


    min_val = attr_to_display.min(-1)[0].unsqueeze(-1).expand_as(attr_to_display)
    max_val = attr_to_display.max(-1)[0].unsqueeze(-1).expand_as(attr_to_display)
    attr_to_display = ((attr_to_display - min_val) / (max_val - min_val+.0000000001))
    attr_to_display = attr_to_display.mean(-2)

    utils.display_parts(args, part_information, top_tiles, risks_to_display,
                        test_subset_library, out_dir, checkpoint['epoch'], attr_to_display)


    risks_out = pd.DataFrame({'SlideID': val_id.long().numpy(),
                              'Duration': duration,
                              'Recurrence': event,
                              '0': risk.squeeze().cpu().detach().numpy()})
    print('Saving to '+os.path.join(out_dir, str(checkpoint['epoch'])+'risks.csv', ))
    risks_out.to_csv(os.path.join(out_dir, str(checkpoint['epoch'])+'risks.csv', ), index=False)




class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = F._Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)

    return indicator_matrix



class testLoss(nn.Module):
    def __init__(self):
        super(testLoss, self).__init__()

    def forward(self, risk_pred, y, e):
        e = e.int().unsqueeze(1).to(gpu)
        y = y.unsqueeze(1)
        mask = torch.ones(y.shape[0], y.shape[0], device=gpu)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)

        return neg_log_loss



class RotateSq:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)



def calculate_centroids(embedding, assignments, centroids):
    for j in range(0, args.n_cluster):
        cluster_j = embedding[(assignments == j).nonzero().squeeze()]
        if len(cluster_j) != 0:
            centroids[j] = cluster_j.mean(0)
    return centroids


def assign(batch_embedding, centroids):
    n = batch_embedding.size(0)
    m = centroids.size(0)
    d = batch_embedding.size(1)
    x = batch_embedding.unsqueeze(1).expand(n, m, d)
    y = centroids.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(torch.pow(x - y, 2).sum(2), .5)
    assignments = dist.argmin(dim=1)
    return assignments



def part_selection(slide_embedding, slide_centroids, slide_assignments):
    selected_part_indices = torch.zeros(args.n_cluster).long()
    c_mask = np.zeros(args.n_cluster, dtype=int)
    for i in range(0, args.n_cluster):
        specific_embedding = slide_embedding[slide_assignments == i]
        if specific_embedding.size(0) > 0:
            specific_centroid = slide_centroids[i].expand(specific_embedding.shape)
            dist = torch.pow(torch.pow(specific_embedding - specific_centroid, 2).sum(1), .5)
            if len(dist) >= args.topk:
                _, topk_indices = dist.topk(k=args.topk, dim=0, largest=False, sorted=True)
                local_idx = topk_indices[torch.randint(0, args.topk, size=(1,))]
                # _, topk_indices = dist.topk(k=2, dim=0, largest=False, sorted=True)
                # local_idx = topk_indices[1]
            else:
                local_idx = dist.argmin()
            nearest = specific_embedding[local_idx]
            selected_part_indices[i] = (slide_embedding == nearest.squeeze()).sum(1).argmax()
            c_mask[i] = 1
        else:
            c_mask[i] = 0
    part_weights = np.asarray([(slide_assignments == x).sum().item() for x in range(args.n_cluster)])
    part_weights = part_weights / part_weights.sum()
    return selected_part_indices, part_weights, c_mask



if __name__ == '__main__':
    main()
