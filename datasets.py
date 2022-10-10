import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from moic.utils import load_json
from PIL import Image
from config import *

from moic.utils import load_json

class GeometricObjectsData(Dataset):
    def __init__(self,split = "train",name = "ccc",resolution = model_opt.resolution):
        super().__init__()
        assert split in ["train","test"],print("split {} not recognized.".format(split))
        self.root_dir = "geoclidean"
        self.concept_name = name
        self.split = split
        self.files = os.listdir(os.path.join(
            self.root_dir,"constraints","concept_{}".format(self.concept_name),
            self.split
        ))
        self.concept_path = os.path.join(
            self.root_dir,"constraints","concept_{}".format(self.concept_name),
            self.split
        )
        self.img_transform = transforms.Compose(
            [   
                transforms.ToTensor()]
        )
        self.question_file = None
        self.resolution = resolution

    def __len__(self):return len(self.files)

    def __getitem__(self,index):
        index = index + 1
        image = Image.open(os.path.join(self.concept_path,"{}_fin.png").format(index)).convert('L')
        image = self.img_transform(image.resize(self.resolution))
        concept_file = open(os.path.join(self.root_dir,"constraints","concept_{}".format(self.concept_name),"concept.txt"), "r")
        programs = concept_file.readlines()
        return {"image":image,"programs":programs}

class GeometricElementsData(Dataset):
    def __init__(self,split = "train",name = "angle",resolution = model_opt.resolution):
        super().__init__()
        assert split in ["train","test"],print("split {} not recognized.".format(split))
        self.root_dir = "geoclidean"
        self.concept_name = name
        self.split = split

        self.files = os.listdir(os.path.join(
            self.root_dir,"elements","concept_{}".format(self.concept_name),
            self.split
        ))
        self.concept_path = os.path.join(
            self.root_dir,"elements","concept_{}".format(self.concept_name),
            self.split
        )
        self.img_transform = transforms.Compose(
            [   
                transforms.ToTensor()]
        )
        self.resolution = resolution

    def __len__(self):return len(self.files)

    def __getitem__(self,index):
        index = index + 1
        image = Image.open(os.path.join(self.concept_path,"{}_fin.png").format(index)).convert('L')
        image = self.img_transform(image.resize(self.resolution))
        concept_file = open(os.path.join(self.root_dir,"elements","concept_{}".format(self.concept_name),"concept.txt"), "r")
        raw_programs = concept_file.readlines()
        programs = [term[1:-3] for term in raw_programs]
        
        return {"image":image,"programs":programs}


class GCLData(Dataset):
    def __init__(self,name = "r1",resolution = model_opt.resolution):
        super().__init__()

        self.root_dir = "geoclidean_framework"
        self.concept_name = name


        self.files = os.listdir(os.path.join(
            self.root_dir,"data","{}".format(self.concept_name),

        ))
        self.concept_path = os.path.join(
            self.root_dir,"data","{}".format(self.concept_name),

        )
        self.img_transform = transforms.Compose(
            [   
                transforms.ToTensor()]
        )
        self.resolution = resolution
        self.params = load_json(os.path.join(
            self.root_dir,"data","{}".format(self.concept_name),
            "params.json"
        ))

    def __len__(self):return len(self.files) - 2

    def __getitem__(self,index):
        index = index + 1
        image = Image.open(os.path.join(self.concept_path,"{}.png").format(index)).convert('L')
        image = self.img_transform(image.resize(self.resolution))

        params = self.params["data"][index]
        
        return {"image":image,"params":params}