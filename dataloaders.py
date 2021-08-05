import os
import torch
import numpy as np
import torch.utils.data
from openslide import OpenSlide


class TileLoader(torch.utils.data.Dataset):
    def __init__(self, args, library,transform):
        data_path = args.slide_path
        unique_ids = library.SlideID.unique()
        print('Loading {} Slides'.format(len(unique_ids)))
        slide_dict = {}
        for i, name in enumerate(unique_ids):
            slide_dict[name] = i
        opened_slides = []
        for name in unique_ids:
            opened_slides.append(OpenSlide(os.path.join(data_path, str(name) + '.svs')))
        self.data = []
        self.opened_slides = opened_slides
        self.slide_dict = slide_dict
        self.transform = transform
        self.len = len(library)
        self.mode = None
        self.local_centroids = None
        self.assignments = None
        self.args = args
    def constraint(self, local_centroids, assignments, library):
        self.data = list(zip(library["SlideID"], library["x"], library["y"], library.index.values))
        self.mode = 'constraint'
        self.local_centroids = local_centroids
        self.assignments = assignments
        self.len = len(library)
    def assignment(self, library):
        self.data = list(zip(library["SlideID"], library["x"], library["y"], library.index.values))
        self.mode = 'assignment'
        self.len = len(self.data)
    def __getitem__(self, index):
        im_id, x, y, local_ind = self.data[index]
        if self.args.level == 0:
            img = self.opened_slides[self.slide_dict[im_id]].read_region([x, y], 0, [256, 256])
        elif self.args.level == 1:
            img = self.opened_slides[self.slide_dict[im_id]].read_region([x, y], 1, [224, 224])
        raw = img.convert('RGB')
        img = self.transform(raw)
        if self.mode == 'constraint':
            slide_centroids = self.local_centroids[im_id]
            cluster = slide_centroids[self.assignments[index].item()]
        else:
            cluster = 0
        return img, cluster, im_id, local_ind, index  # , transforms.ToTensor()(raw)
    def __len__(self):
        return self.len



class AssignmentLoader(torch.utils.data.Dataset):
    def __init__(self, args, library,transform):
        data_path = args.slide_path
        unique_ids = library.SlideID.unique()
        slide_dict = {}
        for i, name in enumerate(unique_ids):
            slide_dict[name] = i
        opened_slides = []
        for name in unique_ids:
            opened_slides.append(OpenSlide(os.path.join(data_path, str(name) + '.svs')))
        self.opened_slides = opened_slides
        self.slide_dict = slide_dict
        self.transform = transform
        self.local_centroids = None
        self.assignments = None
        self.data = list(zip(library["SlideID"], library["x"], library["y"], library.index.values))
        self.mode = 'assignment'
        self.len = len(self.data)
        self.args = args
    def __getitem__(self, index):
        im_id, x, y, local_ind = self.data[index]
        if self.args.level == 0:
            img = self.opened_slides[self.slide_dict[im_id]].read_region([x, y], 0, [256, 256])
        elif self.args.level == 1:
            img = self.opened_slides[self.slide_dict[im_id]].read_region([x, y], 1, [224, 224])
        raw = img.convert('RGB')
        img = self.transform(raw)
        cluster = 0
        return img, cluster, im_id, local_ind, index  # , transforms.ToTensor()(raw)
    def __len__(self):
        return self.len


class ConstraintLoader(torch.utils.data.Dataset):
    def __init__(self, args, library, transform, local_centroids, assignments):
        data_path = args.slide_path
        unique_ids = library.SlideID.unique()
        slide_dict = {}
        for i, name in enumerate(unique_ids):
            slide_dict[name] = i
        opened_slides = []
        for name in unique_ids:
            opened_slides.append(OpenSlide(os.path.join(data_path, str(name) + '.svs')))
        self.opened_slides = opened_slides
        self.slide_dict = slide_dict
        self.transform = transform
        self.data = list(zip(library["SlideID"], library["x"], library["y"], library.index.values, assignments.numpy()))
        self.mode = 'constraint'
        self.local_centroids = local_centroids
        self.len = len(library)
        self.args = args
    def __getitem__(self, index):
        im_id, x, y, local_ind, assignment = self.data[index]
        if self.args.level == 0:
            img = self.opened_slides[self.slide_dict[im_id]].read_region([x, y], 0, [256, 256])
        elif self.args.level == 1:
            img = self.opened_slides[self.slide_dict[im_id]].read_region([x, y], 1, [224, 224])
        raw = img.convert('RGB')
        img = self.transform(raw)
        slide_centroids = self.local_centroids[im_id]
        cluster = slide_centroids[assignment]
        return img, cluster, im_id, local_ind, index  # , transforms.ToTensor()(raw)
    def __len__(self):
        return self.len



class PartLoader(torch.utils.data.Dataset):
    def __init__(self, args, library, top_tiles, transforms, part_information):
        data_path = args.slide_path
        unique_ids = library.SlideID.unique()
        slide_dict = {}
        for i, name in enumerate(unique_ids):
            slide_dict[name] = i
        opened_slides = []
        for name in unique_ids:
            opened_slides.append(OpenSlide(os.path.join(data_path, str(name) + '.svs')))
        survival_data = library.groupby('SlideID').apply(lambda x: x.sample(n=1, replace=True))
        self.data = list(zip(survival_data["SlideID"], survival_data['Duration'], survival_data['Event']))
        self.opened_slides = opened_slides
        self.slide_dict = slide_dict
        self.transform = transforms
        self.top_tiles = top_tiles
        self.library = library
        self.part_information = part_information
        self.len = len(library.SlideID.unique())
        self.args = args
    def __getitem__(self, index):
        im_id, duration, event = self.data[index]
        parts = [None]*self.args.n_cluster
        information = self.part_information[im_id]
        weights, mask = zip(*information)
        part_idx = self.top_tiles[im_id]
        for i in range(0, self.args.n_cluster):
            if mask[i]:
                x = int(self.library.loc[part_idx[i].item()].x)
                y = int(self.library.loc[part_idx[i].item()].y)
                if self.args.level == 0:
                    img = self.opened_slides[self.slide_dict[im_id]].read_region([x, y], 0, [256, 256])
                elif self.args.level == 1:
                    img = self.opened_slides[self.slide_dict[im_id]].read_region([x, y], 1, [224, 224])
                raw = img.convert('RGB')
                img = self.transform(raw)
                parts[i] = img
            else:
                parts[i] = torch.zeros(3, 224, 224)
        return parts, duration, event, np.asarray(weights), np.asarray(mask), im_id, index
    def __len__(self):
        return self.len

