import os
import torch
from openslide import OpenSlide
from torchvision.utils import save_image
import torchvision.transforms as transforms
from pdb import set_trace
from matplotlib import cm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_error(train_ci, val_ci, trainLoss, epoch, slname):
    if epoch == 0:
        f = open(slname, 'w')
        f.write('epoch,trainLoss,trainCI,valCI\n')
        f.write('{},{:.4f},{:.4f},{:.4f}\n'.format(epoch, trainLoss, train_ci, val_ci))
        f.close()
    else:
        f = open(slname, 'a')
        f.write('{},{:.4f},{:.4f},{:.4f}\n'.format(epoch, trainLoss, train_ci, val_ci))
        f.close()


def display(args, cluster_centers, full_code, assigned_clusters, locations, epoch, out_dir):
    total_id = locations[:, 0]
    x_patch = locations[:, 1]
    y_patch = locations[:, 2]
    k = 0
    cols = 20
    grid_batch = torch.randn(args.n_cluster * cols, 3, 224, 224, requires_grad=False)
    for i in range(0, len(cluster_centers)):
        cluster_members = full_code[(assigned_clusters == i).nonzero().squeeze()]
        print(cluster_members.size())
        if cluster_members.size(0) < cols or len(cluster_members.size()) == 1:
            for n in range(0, cols):
                grid_batch[k] = torch.zeros(1, 3, 224, 224)
                k = k + 1
        else:
            data_path = args.slide_path
            cluster_x = x_patch[(assigned_clusters == i).nonzero().squeeze()].numpy()
            cluster_y = y_patch[(assigned_clusters == i).nonzero().squeeze()].numpy()
            cluster_id = total_id[(assigned_clusters == i).nonzero().squeeze()].numpy()
            distances = torch.nn.PairwiseDistance()(cluster_members,
                                                    cluster_centers[i].repeat(cluster_members.size(0), 1))
            rand = distances.topk(cols, largest=True)[1]
            for n in range(0, len(rand)):
                index = rand[n]
                svs = OpenSlide(os.path.join(data_path, str(cluster_id[index]) + '.svs'))
                patch = svs.read_region([cluster_x[index], cluster_y[index]], args.level, [224, 224])
                patch = patch.resize((224, 224)).convert('RGB')
                patch = transforms.ToTensor()(patch)
                grid_batch[k] = patch
                k = k + 1
    save_image(grid_batch, out_dir + '/clustering_grid_top/image_{}.png'.format(epoch), nrow=cols)


def display_parts(args, part_information, top_tiles, risks_to_display, validation_library, out_dir, epoch, attr_to_display):
    magma = cm.get_cmap('PiYG')
    data_path = args.slide_path
    display_tensor = torch.rand(args.n_cluster * len(risks_to_display), 3, 224, 224, requires_grad=False)
    print(risks_to_display)
    k = 0
    for i in range(0, len(risks_to_display)):
        im_id = risks_to_display[i].item()
        information = part_information[im_id]
        weights, mask = zip(*information)
        part_idx = top_tiles[im_id]
        svs = OpenSlide(os.path.join(data_path, str(im_id) + '.svs'))
        for n in range(0, args.n_cluster):
            if mask[n] == 1:
                x = int(validation_library.loc[part_idx[n].item()].x)
                y = int(validation_library.loc[part_idx[n].item()].y)
                img = svs.read_region([x, y], 0, [224, 224])
                display_tensor[k] = transforms.ToTensor()(img.convert('RGB'))
            else:
                display_tensor[k] = torch.zeros(3, 224, 224)
            k = k + 1
    x_transformed = magma(attr_to_display)
    x_transformed = torch.from_numpy(x_transformed)
    x_transformed = x_transformed[:, 0:3]
    colorrow = torch.randn(args.n_cluster, 3, 224, 224)
    for i in range(0, args.n_cluster):
        colorrow[i, 0] = x_transformed[i, 0]
        colorrow[i, 1] = x_transformed[i, 1]
        colorrow[i, 2] = x_transformed[i, 2]
    display_tensor = torch.cat([colorrow, display_tensor])
    print(out_dir + '/part_viz/')
    save_image(display_tensor, out_dir + '/part_viz/image_{}.png'.format(epoch), nrow=args.n_cluster)


def save_checkpoint(state, path, filename):
    fullpath = os.path.join(path, filename)
    torch.save(state, fullpath)



def get_mean_and_std(dataset, library):
    '''Compute the mean and std value of dataset.'''
    bs = 1024
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=10)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    dataloader.dataset.assignment(library)
    counter = 0
    for img, _, _, _, _ in dataloader:
        mean[0] += bs*img[:,0,:,:].mean()
        std[0] += bs*img[:,0,:,:].std()
        mean[1] += bs * img[:, 1, :, :].mean()
        std[1] += bs * img[:, 1, :, :].std()
        mean[2] += bs * img[:, 2, :, :].mean()
        std[2] += bs * img[:, 2, :, :].std()
        counter = counter + 1
        print('[{0}/{1}] \t'.format(counter, len(dataloader)))
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
