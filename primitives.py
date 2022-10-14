import torch
import torch.nn as nn

from moic.data_structure import *
from moic.mklearn.nn.functional_net import FCBlock

import networkx as nx


import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiLineString

from config import *

BCELoss= nn.BCELoss(reduction = "mean")

def ptype(inputs):
    if inputs[0] == "c": return "circle"
    if inputs[0] == "l": return "line" 
    if inputs[0] == "p": return "point"

# geometric structure model. This is used to create the geometric concept graph
# and realize the concept and make sample of concepts. This is practically the decoder part of the 
# Geometric AutoEncoder model

dgc = ["l1 = line(p1(), p2())","c1* = circle(p1(), p2())","c2* = circle(p2(), p1())","l2 = line(p1(), p3(c1, c2))","l3 = line(p2(), p3()))"]

def parse_geoclidean(programs = dgc):
    outputs = []
    for program in programs:
        left,right = program.split("=")
        left = left.replace(" ","");right = right.replace(" ","")
        func_node_form = toFuncNode(right)
        func_node_form.token = left
        outputs.append(func_node_form)
    return outputs

class PointDecoder(nn.Module):
    def __init__(self,in_dim = 512):
        super().__init__()
        self.raw_decoder = FCBlock(132,4,in_dim,2)
        self.scale = 32
    def forward(self,x):return torch.tensor([[32,32]]) + self.scale * self.raw_decoder(x)

class PointProp(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt = opt
        self.update_map   = nn.Linear(opt.latent_dim,opt.latent_dim)
        self.message_map  = nn.Linear(opt.latent_dim,opt.latent_dim)
        self.joint_update = FCBlock(132,2,opt.encoder_dim + opt.latent_dim,opt.latent_dim)

    def forward(self,signal,components):
        if not components: 
            return self.joint_update(torch.cat([signal,torch.zeros([1,self.opt.latent_dim])] ,-1))
        right_inters = 0
        for comp in components:right_inters += self.message_map(comp)

        right_inters = self.update_map(right_inters)
        return self.joint_update(torch.cat([signal,right_inters],-1))

class MessageProp(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.update_map   = nn.Linear(opt.latent_dim,opt.latent_dim)
        self.message_map  = nn.Linear(opt.latent_dim,opt.latent_dim)
        self.joint_update = FCBlock(132,2,2 *  opt.latent_dim,opt.latent_dim)
        self.opt = opt

    def forward(self,signal,components):
        if not components: 
            return self.joint_update(torch.cat([signal,torch.zeros([1,self.opt.latent_dim])] ,-1))
        right_inters = 0
        for comp in components:right_inters += self.message_map(comp)

        right_inters = self.update_map(right_inters)
        return self.joint_update(torch.cat([signal,right_inters],-1))

def find_connection(node,graph,loc = 0):
    outputs = []
    for edge in graph.edges:
        if edge[loc] == node:outputs.append(edge[int(not loc)])
    return outputs

class GeometricConstructorLegacy(nn.Module):
    def __init__(self,opt = model_opt):
        super().__init__()

        self.realized = False
        self.structure = None
        self.visisble = []
        self.global_feature = None
        self.mloss = 0 # loss of the mask prediction
        self.ploss = 0 # loss of the point decoder
        self.constrution_logp = 0 # the log prob of a configuruation is created.

        # this is the feature propagator for upward and downward quest
        self.line_propagator = FCBlock(132,3,opt.latent_dim * 2, opt.latent_dim)
        self.circle_propagator  = FCBlock(132,3,opt.latent_dim * 2, opt.latent_dim)
        self.point_propagator   = PointProp(opt)
        self.message_propagator = MessageProp(opt)

        # graph propagation [positional encoding] storage
        self.upward_memory   = None
        self.downward_memory = None

        # local and global encoders
        self.query_encoder = nn.Transformer(nhead=16, num_encoder_layers=12)
        self.global_encoder = FeatureEncoder(input_nc = 1,z_dim = opt.latent_dim)
        self.gauge = nn.Linear(64 * 64 * opt.latent_dim,512)
        self.local_encoder = None
        self.local_decoder = FeatureDecoder(512)
        self.point_decoder = PointDecoder(512)

        # store some basic configs
        self.resolution = opt.resolution
        self.opt = opt

        self.clear()

    def clear(self):
        self.realized  = False # clear the state of dag and the realization
        self.structure = None  # clear the state of cocnept structure 
        self.global_feature = None # clear the encoder feature on the input image
        self.upward_memory   = None
        self.downward_memory = None 

        self.mloss = 0 # this is the loss for the mask prediction
        self.ploss = 0 # this is the loss for the single point
        

    def make_dag(self,concept_struct):
        """
        input:  the concept struct is a list of func nodes
        output: make self.struct as a list of nodes and edges
        """
        if isinstance(concept_struct[0],str):concept_struct = parse_geoclidean(concept_struct)
        realized_graph  = nx.DiGraph()
        self.visible = []

        def parse_node(node):
            node_name = node.token
            
            # if the object is already in the graph, jsut return the name of the concept
            if node_name in realized_graph.nodes: return node_name
            if node_name == "":# if this place is a void location.
                node_name = "<V>";visible = False
            elif node_name[-1] == "*":
                node_name = node_name.replace("*","");visible =False
            else:visible = True
            realized_graph.add_node(node_name)
            for child in node.children:
                if visible:self.visible.append(node_name)
                realized_graph.add_edge(parse_node(child),node_name) # point from child to current node
    
            return node_name

        for program in concept_struct:parse_node(program)
        self.structure = realized_graph
        self.realized = True
    
        return realized_graph

    def realize(self,signal):
        # given every node a vector representation
        # 1. start the upward propagation
        upward_memory_storage   = {}
        def quest_down(node):
            if node in upward_memory_storage:return upward_memory_storage[node]# is it is calculated, nothing happens
            primitive_type =  ptype(node)
            connect_to     =  find_connection(node,self.structure,loc = 1)
            if node == "<V>":return
            if primitive_type == "circle": # use the circle propagator to calculate mlpc(cat([ec1,ec2]))
                assert len(connect_to) == 2,print("the circle is connected to {} parameters (2 expected).".format(len(connect_to)))
                left_component   = quest_down(connect_to[0]);right_component  = quest_down(connect_to[1])
                update_component = self.circle_propagator(torch.cat([left_component,right_component],-1))
            if primitive_type == "line":
                assert len(connect_to) == 2,print("the line is connected to {} parameters (2 expected).".format(len(connect_to)))
                start_component  = quest_down(connect_to[0]);end_component    = quest_down(connect_to[1])
                update_component = self.line_propagator(torch.cat([start_component,end_component],-1))
            if primitive_type == "point":
                point_prop_inputs = []
                for component in connect_to:
                    if component != "<V>":point_prop_inputs.append(quest_down(component)) # the input prior is the intersection of some component
                update_component = self.point_propagator(signal,point_prop_inputs)
        
            upward_memory_storage[node] = update_component 
            return update_component
        
        for node in self.structure.nodes:quest_down(node)
        # update the memory unit after the propagation
        self.upward_memory = upward_memory_storage

        # 2. start the downward propagation. (maybe not)
        downward_memory_storage   = {}
        def quest_up(node):
            if node == "<V>":return
            if node in downward_memory_storage:return downward_memory_storage[node]# if node already calculated, nothing happens
            connect_to     =  find_connection(node,self.structure,loc = 0) # find all the nodes that connected to the current node

            input_neighbors = [quest_up(p_node) for p_node in connect_to]
            current_node_feature = self.upward_memory[node] # this is the feature a point store currently (circle,point,line aware)
            update_component = self.message_propagator(current_node_feature,input_neighbors) # this is the update component feature
        
            downward_memory_storage[node] = update_component 
            return update_component
        for node in self.structure: quest_up(node)

        # update the memory unit of the propagation
        self.downward_memory = downward_memory_storage
        return 

    def construct(self,lcnet):
        # lcnet provides a set of lines and circles with embeddings
        lines   = lcnet.lines # embeddings with a diction
        circles = lcnet.circles # embeddings with a diction

        realized_visibles = []
        def build_node(node):
            if node not in self.visible:return
            if ptype(node) == "point":return
        
        for node in self.structure.nodes:build_node(node)
        

    def construct_legacy(self,image = None,target = None):
        plt.cla()
        calculated_node = {}

        all_view_objects =[]

        output_image = 0
        def build(node):
            if node in calculated_node:return calculated_node[node]
            if node == "<V>":return 0
            node_type = ptype(node)
            u_feature,d_feature = self.upward_memory[node],self.downward_memory[node]
            node_feature = torch.cat([u_feature,d_feature],-1)

            connect_to = find_connection(node,self.structure,1)
            
            query_feature = self.query_encoder(self.global_feature,node_feature)
            if node_type == "circle": # suppose to decode a mask <U,D> -> T[E] -> M
                p1 = build(connect_to[0]);p2 = build(connect_to[1])
                point_center = Point(p1[0][0],p1[0][1])
                point_c = Point(p2[0][0],p2[0][1])
                radius = point_center.distance(point_c)
                circle = Point(point_center.x, point_center.y).buffer(radius)
                
                if node in self.visible:all_view_objects.append(circle)
                return 0
            if node_type == "line": # suppose to decode a mask <U,D> -> T[E] -> M
                p1 = build(connect_to[0]);p2 = build(connect_to[1])
                point_a = Point(p1[0][0],p1[0][1])
                point_b = Point(p2[0][0],p2[0][1])
                
                if node in self.visible:all_view_objects.append(LineString([(point_a.x, point_a.y), (point_b.x, point_b.y)]))
                return 0
            if node_type == "point": # suppose to decode a point <U,D> -> T[E] -> (x,y)
                decode_point = self.point_decoder(query_feature)
                calculated_node[node] = decode_point
                point_target = target[node]
                point_wise_loss = torch.nn.functional.mse_loss(decode_point[0].float(),torch.tensor(point_target).float())
                print(point_wise_loss)

                self.ploss += point_wise_loss
                return decode_point
            calculated_node[node] = 1
            return 0
        
        for node in self.structure.nodes:build(node)

        plt.figure("inputs vs recons")
        plt.subplot(122);plt.cla();
        for obj in all_view_objects:plot_object(obj)
        plt.pause(0.001)
        
        return 

    def train(self,x,concept = None,target_dag = None):
        feature_encode = self.global_encoder(x).flatten(start_dim = 1)

        self.global_feature = self.gauge(feature_encode)

        self.make_dag(concept)
        self.realize(torch.randn([1,128]))

        # do something with the decoder
        
        return self.construct(x,target_dag)

    def forward(self,x,concept = None,target_dag = None):
        
        feature_encode = self.global_encoder(x)
        self.make_dag(concept)
        self.realize()

        # do something with the decoder
        self.constuct(target_dag,x)
        
        return x

class GeometricConstructor(nn.Module):
    def __init__(self,opt = model_opt):
        super().__init__()

        self.realized = False
        self.structure = None
        self.visisble = []
        self.global_feature = None
        self.construction_logp = 0 # the log prob of a configuruation is created.

        # this is the feature propagator for upward and downward quest
        self.line_propagator = FCBlock(132,3,opt.latent_dim * 2, opt.latent_dim)
        self.circle_propagator  = FCBlock(132,3,opt.latent_dim * 2, opt.latent_dim)
        self.point_propagator   = PointProp(opt)
        self.message_propagator = MessageProp(opt)

        # graph propagation [positional encoding] storage
        self.upward_memory   = None
        self.downward_memory = None

        # store some basic configs
        self.resolution = opt.resolution
        self.opt = opt

        self.clear()

    def clear(self):
        self.realized  = False # clear the state of dag and the realization
        self.structure = None  # clear the state of cocnept structure 
        self.global_feature = None # clear the encoder feature on the input image
        self.upward_memory   = None
        self.downward_memory = None         

    def build_dag(self,concept_struct):
        """
        input:  the concept struct is a list of func nodes
        output: make self.struct as a list of nodes and edges
        """
        if isinstance(concept_struct[0],str):concept_struct = parse_geoclidean(concept_struct)
        realized_graph  = nx.DiGraph()
        self.visible = []

        def parse_node(node):
            node_name = node.token
            
            # if the object is already in the graph, jsut return the name of the concept
            if node_name in realized_graph.nodes: return node_name
            if node_name == "":# if this place is a void location.
                node_name = "<V>";visible = False
            elif node_name[-1] == "*":
                node_name = node_name.replace("*","");visible =False
            else:visible = True
            realized_graph.add_node(node_name)
            for child in node.children:
                if visible:self.visible.append(node_name)
                realized_graph.add_edge(parse_node(child),node_name) # point from child to current node
    
            return node_name

        for program in concept_struct:parse_node(program)
        self.structure = realized_graph
        self.realized = True
    
        return realized_graph

    def realize(self,signal):
        # given every node a vector representation
        # 1. start the upward propagation
        upward_memory_storage   = {}
        def quest_down(node):
            if node in upward_memory_storage:return upward_memory_storage[node]# is it is calculated, nothing happens
            primitive_type =  ptype(node)
            connect_to     =  find_connection(node,self.structure,loc = 1)
            if node == "<V>":return
            if primitive_type == "circle": # use the circle propagator to calculate mlpc(cat([ec1,ec2]))
                assert len(connect_to) == 2,print("the circle is connected to {} parameters (2 expected).".format(len(connect_to)))
                left_component   = quest_down(connect_to[0]);right_component  = quest_down(connect_to[1])
                update_component = self.circle_propagator(torch.cat([left_component,right_component],-1))
            if primitive_type == "line":
                assert len(connect_to) == 2,print("the line is connected to {} parameters (2 expected).".format(len(connect_to)))
                start_component  = quest_down(connect_to[0]);end_component    = quest_down(connect_to[1])
                update_component = self.line_propagator(torch.cat([start_component,end_component],-1))
            if primitive_type == "point":
                point_prop_inputs = []
                for component in connect_to:
                    if component != "<V>":point_prop_inputs.append(quest_down(component)) # the input prior is the intersection of some component
                update_component = self.point_propagator(signal,point_prop_inputs)
        
            upward_memory_storage[node] = update_component 
            return update_component
        
        for node in self.structure.nodes:quest_down(node)
        # update the memory unit after the propagation
        self.upward_memory = upward_memory_storage

        # 2. start the downward propagation. (maybe not)
        downward_memory_storage   = {}
        def quest_up(node):
            if node == "<V>":return
            if node in downward_memory_storage:return downward_memory_storage[node]# if node already calculated, nothing happens
            connect_to     =  find_connection(node,self.structure,loc = 0) # find all the nodes that connected to the current node

            input_neighbors = [quest_up(p_node) for p_node in connect_to]
            current_node_feature = self.upward_memory[node] # this is the feature a point store currently (circle,point,line aware)
            update_component = self.message_propagator(current_node_feature,input_neighbors) # this is the update component feature
        
            downward_memory_storage[node] = update_component 
            return update_component
        for node in self.structure: quest_up(node)

        # update the memory unit of the propagation
        self.downward_memory = downward_memory_storage
        return 
    
    def construct(self,lines,circles,mode = "train"):
        # lcnet provides a set of lines and circles with embeddings
        line_features   = lines # embeddings with a diction
        circle_features = circles # embeddings with a diction

        realized_visibles = []
        def build_node(node):
            if node not in self.visible:return
            if ptype(node) == "point":return
            feature = torch.cat([self.upward_memory[node],self.downward_memory[node]],-1)
            if ptype(node) == "line":
                if mode == "train":
                    choice,p = make_pdf(feature,line_features);self.construction_logp += torch.log(p)
                    #line_params = lines[choice]
                else:
                    choice = make_pdf(feature,line_features);self.construction_logp += torch.log(p)
                    #line_params = lines[choice]
            if ptype(node) == "circle":
                if mode == "train":
                    choice,p = make_pdf(feature,circle_features);self.construction_logp += torch.log(p)
                    #circle_params = circles[choice]
                else:
                    choice,p = make_pdf(feature,circle_features);self.construction_logp += torch.log(p)
                    #circle_params = circles[choice]
        for node in self.structure.nodes:build_node(node)
        return 0,self.construction_logp

def make_pdf(source,choices,mode = "random"):
    features = torch.cat([k.unsqueeze(0) for k in choices[1]],0)

    keys = choices[0]

    pdf = torch.softmax(torch.cosine_similarity(features,source)* 5,0)
    #print(pdf)
    if mode == "random":
        index = np.random.choice(range(len(keys)),p = pdf.detach().numpy())
        return keys[index],pdf[index]
    else:
        index = np.argmax(pdf.detach().numpy())
        return keys[index],pdf[index]

def numpy_from_plot(ax):
    ax.figure.canvas.draw()
    data = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = ax.figure.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im

def plot_object(obj, color="black"):
    if isinstance(obj, Polygon):
        obj = obj.exterior
    x, y = obj.xy
    plt.plot(x, y, linewidth=3, color=color)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dgc = ["l1 = line(p1(), p2())","c1* = circle(p1(), p2())","c2* = circle(p2(), p1())","l2 = line(p1(), p3(c1, c2))","l3 = line(p2(), p3()))"]

    model = GeometricConstructor(model_opt)
    #outputs = model.train(torch.randn([1,1,64,64]),concept = dgc)
    #print(outputs.shape)
    model.make_dag(dgc)
    model.realize(torch.zeros([1,128]))
    g = model.structure
    nx.draw_networkx(g)
    plt.show()
    
    model.realize(torch.randn([1,128]))