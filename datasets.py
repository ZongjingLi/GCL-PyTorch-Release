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
        image = self.img_transform(image)
        concept_file = open(os.path.join(self.root_dir,"constraints","concept_{}".format(self.concept_name),"concept.txt"), "r")
        programs = concept_file.readlines()
        return {"image":image,"concept":programs,"path":os.path.join(self.root_dir,"constraints","concept_{}".format(self.concept_name),"train/{}_fin.png".format(index))}

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
        image = self.img_transform(image)
        concept_file = open(os.path.join(self.root_dir,"elements","concept_{}".format(self.concept_name),"concept.txt"), "r")
        raw_programs = concept_file.readlines()
        programs = [term[1:-3] for term in raw_programs]
        
        return {"image":image,"concept":programs,"path":os.path.join(self.root_dir,"elements","concept_{}".format(self.concept_name),"train/{}_fin.png".format(index))}


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

    def __len__(self):return 30 # len(self.files) 

    def __getitem__(self,index):
        index = index + 1
        image = Image.open(os.path.join(self.concept_path,"{}.png").format(index)).convert('L')
        image = self.img_transform(image.resize(self.resolution))

        params = self.params["data"][index]
        
        return {"image":image,"params":params}



class AllElementsData(Dataset):
    def __init__(self,split = "train",resolution = model_opt.resolution):
        super().__init__()

        elements = ["ang_bisector","angle","diameter","eq_t","oblong","parallel_l",
                    "perp_bisector","quadrilateral","radii","rectilinear","rhomboid",
                    "rhombus","right_ang_t","segment","sixty_ang","square","triangle"]

        assert split in ["train","test"],print("split {} not recognized.".format(split))
        self.root_dir = "geoclidean"
        self.split = split

        self.element_files = []

        for name in elements:
            concept_file = open(os.path.join(self.root_dir,"elements","concept_{}".format(name),"concept.txt"), "r")
            raw_programs = concept_file.readlines()
            program = [term[1:-3] for term in raw_programs]
            for i in range(5):
                name_path = os.path.join(
                self.root_dir,"elements","concept_{}".format(name),
                self.split,"{}_fin.png".format(i+1)
                )
                self.element_files.append([program,name_path])

        self.img_transform = transforms.Compose(
            [   
                transforms.ToTensor()]
        )
        self.resolution = resolution

    def __len__(self):return len(self.element_files)

    def __getitem__(self,index):
        bind =  self.element_files[index]
        image = Image.open(bind[1]).convert('L')
        image = self.img_transform(image)
        return {"concept":bind[0],"image":image,"path":bind[1]}