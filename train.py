import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='EPIC-Survival')
parser.add_argument('--lr', default=0.1, type=float, metavar='float', help='learning rate')
parser.add_argument('--dropout', default=.5, type=float, metavar='0 to 1', help='dropout rate for fc layers')
parser.add_argument('--out', default=1, type=int, metavar='0 or 1', help='0 for testing, 1 to save experiments')
parser.add_argument('-n', '--n-cluster', default=10, type=int, metavar='int', help='# of clusters')
parser.add_argument('-w', '--waist', default=64, type=int, metavar='int', help='encoder length')
parser.add_argument('--development', default=0, type=int, metavar='0 or 1', help='for quick testing')
parser.add_argument('-e', '--epochs', default=3000, type=int, metavar='int', help='# training epochs')
parser.add_argument('--train_full', default=0, type=int, metavar='0 or 1', help='0 for testing on validation')
parser.add_argument('--p-batch-size', default=32, type=int, metavar='int', help='part loader batch size')
parser.add_argument('--t-batch-size', default=1024, type=int, metavar='int', help='tile loader batch size')
parser.add_argument('-s', '--sample', default=100, type=int, metavar='int', help='# tiles sampled per slide')
parser.add_argument('--verbose', default=0, type=int, metavar='0 or 1', help='set to 1 to print extra training details')
parser.add_argument('-k', '--topk', default=3, type=int, metavar='int', help='# top tiles to sample from for part')
parser.add_argument('--subsample', default=1, type=float, metavar='<1', help='percent slides subsampled per epoch')
parser.add_argument('--clusteringlambda', default=1, type=float, metavar='0 to 1', help='constraint weight in loss')
parser.add_argument('--weightdecay', default=1e-4, type=float, metavar='0 to 1', help='weight decat')
parser.add_argument('--level', default=0, type=int, metavar='0 or 1', help='tile resolution')
parser.add_argument('--decay_interval', default=0, type=int, metavar='0 or 1', help='epochs between lr drop')
parser.add_argument('--slide_path', default='./', type=str, metavar='dir', help='path to svs directory')
parser.add_argument('--library', default='./', type=str, metavar='file', help='path to tile library csv')

import os
import utils
import torch
import dataloaders
import numpy as np
import torch.nn as nn
from loss import NLPLLossStrat
import torchvision.transforms as transforms
from models import resblock18, SurvivalResNet
from lifelines.utils import concordance_index

from pdb import set_trace


def main():
    print('!! -------- Initializing EPIC-Survival -------- !!')
    global args, gpu, best_train_ci, best_val_ci
    best_train_ci = 0
    best_val_ci = 0
    args = parser.parse_args()

    gpu = torch.device('cuda')
    root_output = './'#change
    if args.out == 0:
        out_dir = os.path.join(root_output, 'test')
    elif args.out == 1:
        save_path = '{}lr_{}d_{}n_{}p_{}w_{}b_{}wd'.format(args.lr,
                                                           args.dropout,
                                                           args.n_cluster,
                                                           args.topk,
                                                           args.waist,
                                                           args.p_batch_size,
                                                           args.weightdecay,
                                                            )
        if args.train_full == 1:
            save_path = save_path + '/test'
        out_dir = os.path.join(root_output, save_path)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'clustering_grid_top')):
        os.mkdir(os.path.join(out_dir, 'clustering_grid_top'))
    if not os.path.exists(os.path.join(out_dir, 'part_viz')):
        os.mkdir(os.path.join(out_dir, 'part_viz'))

    ###################################################################################################################

    print('Initialization [1/4] ........ Building Model Architecture')

    feature_extractor = resblock18(args.waist)
    feature_extractor = nn.DataParallel(feature_extractor).to(gpu)
    model = SurvivalResNet(feature_extractor, args.waist, num_clusters=args.n_cluster, dropout=args.dropout).to(gpu)

    criterion = NLPLLossStrat()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    sl = args.decay_interval
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[sl, 2 * sl, 3 * sl, 4 * sl], gamma=0.1)
    workers = 10

###################################################################################################################

    print('Initialization [2/4] ........ Loading Coordinate File and Creating Training/Validation Split')

    #data = pd.read_csv(args.library)
    data = torch.load('/lila/data/fuchs/hassan/cholangio/classiceplcholangio_library.pth', encoding='latin1')['library']
    data = data.astype({"SlideID": int})
    if args.train_full == 1:
        train_library = data[data.Split != 'test'].reset_index(drop=True)
        validate = 0
    else:
        train_library = data[data.Split == 'train'].reset_index(drop=True)
        validation_library = data[data.Split == 'val'].reset_index(drop=True)
        validate = 1
        if args.development == 1:
            train_library = train_library.iloc[0:300000] #subset for quick testing

    ###################################################################################################################

    print('Initialization [3/4] ........ Staging Tiles into Dataloader')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    augmentations = transforms.Compose([transforms.RandomCrop(224),
                                        transforms.ColorJitter(.1, .1, .1, .05),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        utils.RotateSq(angles=[-180, -90, 0, 90, 180]),
                                        transforms.ToTensor(),
                                        normalize
                                        ])



    augmentations_valid = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    tile_dataset = dataloaders.TileLoader(args, train_library, augmentations)
    tile_loader = torch.utils.data.DataLoader(tile_dataset,
                                              batch_size=args.t_batch_size,
                                              shuffle=False,
                                              num_workers=workers,
                                              pin_memory=True)
    if args.train_full == 0:
        tile_dataset_validation = dataloaders.TileLoader(args, validation_library, augmentations_valid)
        tile_loader_validation = torch.utils.data.DataLoader(tile_dataset_validation,
                                                             batch_size=args.t_batch_size,
                                                             shuffle=False,
                                                             num_workers=workers,
                                                             pin_memory=True)

    ###################################################################################################################

    print('Initialization [4/4] ........ Initializing Data Tensors')
    global_centroids = torch.randn(args.n_cluster, args.waist, device=gpu)
    global local_centroids, top_tiles, part_information, assignments
    local_centroids = dict()
    top_tiles = dict()
    part_information = dict()
    assignments = dict()
    b_s = args.t_batch_size
    ###################################################################################################################

    print('!! -------- Training EPIC-Survival -------- !!')
    for epoch in range(0, args.epochs):
        # Subset Training Set for Epoch
        subset = np.random.choice(train_library.SlideID.unique(),
                                  int(len(train_library.SlideID.unique()) * args.subsample), replace=False)
        subset_library = train_library[train_library.SlideID.isin(subset)]
        subset_library = subset_library.groupby('SlideID').apply(lambda x: x.sample(n=args.sample, replace=False))
        subset_library = subset_library.sample(frac=1).droplevel(0)
        tile_loader.dataset.assignment(subset_library)

        # Extract Embedding and Assign Clusters to Tiles for given subset
        global_centroids, subset_assignments, subset_ids, subset_embedding, subset_indices = \
            tile_assign(tile_loader, feature_extractor, subset, global_centroids, epoch)

        # Calculate Slide-Level Centroids, Choose Parts
        slide_to_parts(subset_ids, subset_embedding, subset_assignments, subset_indices, epoch)

        # Creating Part Dataset for Survival Training
        part_dataset = dataloaders.PartLoader(args, subset_library, top_tiles, augmentations, part_information)
        part_loader = torch.utils.data.DataLoader(part_dataset,
                                                  batch_size=args.p_batch_size,
                                                  shuffle=True,
                                                  num_workers=workers,
                                                  pin_memory=True)
        constraint_dataset = dataloaders.ConstraintLoader(args, subset_library, augmentations, local_centroids, subset_assignments)
        constraint_loader = torch.utils.data.DataLoader(constraint_dataset,
                                                        batch_size=256,
                                                        shuffle=False,
                                                        num_workers=workers,
                                                        pin_memory=True)
        #Survival Training
        lossMeter = utils.AverageMeter()
        ciMeter = utils.AverageMeter()
        model.train()
        print('Epoch: {}'.format(epoch))
        for (parts, duration, event, weights, mask, _, index), (constraint, target, _, _, _) \
                in zip(part_loader, constraint_loader):
            img = [x.to(gpu) for x in parts]
            constraint_embedding = feature_extractor(constraint.to(gpu))
            risk, _, _ = model(img, mask, weights, True)
            optimizer.zero_grad()
            ordered_risks, order_idx = torch.sort(risk, 0)
            low = risk[order_idx[(ordered_risks < torch.median(ordered_risks))]]
            high = risk[order_idx[(ordered_risks >= torch.median(ordered_risks))]]
            loss = criterion(-risk, duration, event, constraint_embedding, target, low, high, gpu, args)
            loss.backward()
            optimizer.step()
            lossMeter.update(loss.item(), img[0].size(0))
            ci = concordance_index(duration.numpy(), risk.detach().cpu().numpy(), event.numpy())
            ciMeter.update(ci.item(), img[0].size(0))
            print('Loss {:.3f} \t CI {:.3f}'.format(lossMeter.avg, ciMeter.avg))
        print('Training Corcordance Index...{:.3f}'.format(ciMeter.avg))
        scheduler.step()


            ###################################################################################################################

        if validate == 1 and epoch % 5 == 0:
            print('!! -------- Validating EPIC-Survival -------- !!')
            # Subset Validation Set for Epoch
            validation_set = validation_library.SlideID.unique()
            val_subset_library = validation_library.groupby('SlideID').apply(lambda x: x.sample(n=1000, replace=False))
            val_subset_library = val_subset_library.sample(frac=1).droplevel(0)

            # Initializing Subset Tensors
            validation_embedding = torch.randn(len(validation_set) * 1000, args.waist)
            validation_ids = torch.randint(0, args.n_cluster, (len(validation_set) * 1000,))
            validation_indices = torch.randint(0, args.n_cluster, (len(validation_set) * 1000,))
            validation_assignments = torch.randint(0, args.n_cluster, (len(validation_set) * 1000,))



            # Measure Embedding and Assign Clusters to Tiles
            tile_loader_validation.dataset.assignment(val_subset_library)
            model.eval()
            with torch.no_grad():  # pass subset tiles through base model to retrieve embeddings
                for i, (input, _, id, local_ind, index) in enumerate(tile_loader_validation):
                    output = feature_extractor(input.to(gpu))
                    validation_ids[(i * b_s):(i * b_s + len(input))] = id
                    validation_indices[(i * b_s):(i * b_s + len(input))] = local_ind
                    emb = output.detach()
                    validation_embedding[(i * b_s):(i * b_s + len(input))] = emb
                    batch_assignments = assign(emb, global_centroids)
                    validation_assignments[
                    (i * b_s):(i * b_s + len(input))] = batch_assignments
                    if args.verbose == 1:
                        print('Epoch: [{0}][{1}/{2}] \t'.format(epoch, i, len(tile_loader_validation)))

            # Calculate Slide-Level Centroids, Choose Parts
            for slide in validation_ids.unique():
                slide_embedding = validation_embedding[(validation_ids == slide).nonzero().squeeze()]
                slide_assignments = validation_assignments[(validation_ids == slide).nonzero().squeeze()]
                slide_indices = validation_indices[(validation_ids == slide).nonzero().squeeze()]
                if epoch == 0:
                    slide_centroids = torch.randn(args.n_cluster, args.waist)
                else:
                    if slide.item() in local_centroids:
                        slide_centroids = local_centroids[slide.item()]
                    else:
                        slide_centroids = torch.randn(args.n_cluster, args.waist)
                slide_centroids = calculate_centroids(slide_embedding, slide_assignments, slide_centroids)
                local_centroids[slide.item()] = slide_centroids
                part_indices, part_weights, c_mask = part_selection(slide_embedding, slide_centroids,slide_assignments)
                top_tiles[slide.item()] = slide_indices[part_indices]
                assignments[slide.item()] = slide_assignments[part_indices]
                part_information[slide.item()] = list(zip(part_weights, c_mask))
            part_dataset_validation = dataloaders.PartLoader(args, validation_library,
                                                             top_tiles, augmentations_valid, part_information)
            part_loader_validation = torch.utils.data.DataLoader(part_dataset_validation,
                                                                 batch_size=args.p_batch_size,
                                                                 shuffle=False,
                                                                 num_workers=workers,
                                                                 pin_memory=True)

            #Survival Data Validation
            ciValMeter = utils.AverageMeter()
            model.eval()
            with torch.no_grad():
                for i, (parts, duration, event, weights, mask, val_id, index) in enumerate(
                        part_loader_validation):
                    img = [x.to(gpu) for x in parts]
                    risk, hx, _ = model(img, mask, weights, True)
                    ci = concordance_index(duration.numpy(), risk.detach().cpu().numpy(), event.numpy())
                    ciValMeter.update(ci.item(), img[0].size(0))
            print('Validation Corcordance Index...{:.3f}'.format(ciValMeter.avg))

            #save risk scores every 5 epochs
            if epoch == 5:
                risks_out = pd.DataFrame({'SlideID': val_id.long().numpy(),
                                          'Duration': duration,
                                          'Recurrence': event,
                                          '0': risk.squeeze().cpu().numpy()})


            epoch_ci_val = np.abs(ciValMeter.avg - .5) + .5
            if epoch_ci_val > best_val_ci and epoch > 300:
                print('Saving checkpoint....')
                risks_out[str(epoch)] = risk.cpu().numpy()
                risks_out.to_csv(os.path.join(out_dir, 'risks.csv', ), index=False)
                best_val_ci = epoch_ci_val
                utils.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'g_centers': global_centroids,
                    'l_centers': local_centroids,
                    'top_tiles': top_tiles,
                    'part_information': part_information,
                    'assignments': assignments,
                }, out_dir, 'checkpoint_best.pth')
                risks_out[str(epoch)] = risk.cpu().numpy()

            if epoch > 300 and epoch % 100 == 0:
                print('Saving checkpoint....')
                utils.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'g_centers': global_centroids,
                    'l_centers': local_centroids,
                    'top_tiles': top_tiles,
                    'part_information': part_information,
                    'assignments': assignments,
                }, out_dir, 'checkpoint_{}.pth'.format(epoch))

        epoch_ci = np.abs(ciMeter.avg - .5) + .5
        if args.train_full == 1:
            utils.save_error(epoch_ci, 0, lossMeter.avg, epoch, os.path.join(out_dir, 'convergence.csv'))
        else:
            utils.save_error(epoch_ci, epoch_ci_val, lossMeter.avg, epoch, os.path.join(out_dir, 'convergence.csv'))


                ###################################################################################################################

        print('Updating Centroids................')
        # calculate the new code of the old sample
        tile_loader.dataset.assignment(subset_library)
        full_code_old = torch.zeros(len(subset) * args.sample, args.waist,
                                    requires_grad=False)  # (n_samples, n_features)
        model.eval()
        # feature_extractor.eval()
        locations = torch.LongTensor(len(subset_library), 3)
        with torch.no_grad():
            for i, (input, _, _, _, index) in enumerate(tile_loader):
                batch_data = subset_library.iloc[index]
                slide = torch.from_numpy(batch_data.SlideID.values)
                x = torch.from_numpy(batch_data.x.values.astype(int))
                y = torch.from_numpy(batch_data.y.values.astype(int))
                locations[index] = torch.stack([slide, x, y], 1)
                input = input.to(gpu)
                code = feature_extractor(input)
                full_code_old[(i * args.t_batch_size):(i * args.t_batch_size + len(input))] = code.cpu()
        del input
        # calcualte new centers based on old sample new code and old assignment

        if ciMeter.avg > best_train_ci:
            best_train_ci = ciMeter.avg
        if epoch % 50 == 0 and epoch != 0:
            utils.display(args, global_centroids.cpu(), subset_embedding, subset_assignments, locations, epoch, out_dir)

        global_centroids = calculate_centroids(full_code_old, subset_assignments, global_centroids)


        if args.train_full == 1 and epoch % 50 == 0:
            print('Saving checkpoint....')
            utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'g_centers': global_centroids,
                'l_centers': local_centroids,
                'top_tiles': top_tiles,
                'part_information': part_information,
                'assignments': assignments,
            }, out_dir, 'checkpoint_{}.pth'.format(epoch))





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


def tile_assign(tile_loader, feature_extractor, subset, global_centroids, epoch):
    subset_embedding = torch.randn(len(subset) * args.sample, args.waist)
    subset_ids = torch.randint(0, args.n_cluster, (len(subset) * args.sample,))
    subset_indices = torch.randint(0, args.n_cluster, (len(subset) * args.sample,))
    b_s = args.t_batch_size
    feature_extractor.eval()
    if epoch < 1:
        subset_assignments = torch.randint(0, args.n_cluster, (len(subset) * args.sample,))
    with torch.no_grad():  # pass subset tiles through base model to retrieve embeddings
        for i, (input, _, id, local_ind, index) in enumerate(tile_loader):
            output = feature_extractor(input.to(gpu))
            subset_ids[(i * b_s):(i * b_s + len(input))] = id
            subset_indices[(i * b_s):(i * b_s + len(input))] = local_ind
            emb = output.detach()
            subset_embedding[(i * b_s):(i * b_s + len(input))] = emb
            if epoch >= 1:
                batch_assignments = assign(emb, global_centroids)
                subset_assignments[(i * b_s):(i * b_s + len(input))] = batch_assignments
            if args.verbose == 1:
                print('Epoch: [{0}][{1}/{2}] \t'.format(epoch, i + 1, len(tile_loader)))
    if epoch < 1:
        global_centroids = calculate_centroids(subset_embedding, subset_assignments, global_centroids)
    return global_centroids, subset_assignments, subset_ids, subset_embedding, subset_indices


def slide_to_parts(subset_ids, subset_embedding, subset_assignments, subset_indices, epoch):
    for slide in subset_ids.unique():
        slide_embedding = subset_embedding[(subset_ids == slide).nonzero().squeeze()]
        slide_assignments = subset_assignments[(subset_ids == slide).nonzero().squeeze()]
        slide_indices = subset_indices[(subset_ids == slide).nonzero().squeeze()]

        if epoch == 0:
            slide_centroids = torch.randn(args.n_cluster, args.waist)
        else:
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

if __name__ == '__main__':
    main()
