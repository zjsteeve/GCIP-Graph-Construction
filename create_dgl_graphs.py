from gurobipy import *
import numpy as np
import os
import torch
import dgl
from dgl.data.utils import save_graphs
import math



# Numeric labels for each instance class
# FIXME: You will need to tailor this process to your application
labelMap = {'vrptw': 0, 'cwlp-ss': 1, 'pcp': 2, 'scheduling': 3, 'cvrp': 4, 'clp': 5, 'inr': 6, 'lotsizing': 7, 'coloring': 8, 'tup': 9, 'bppif': 10, 'maplabeling': 11, 'kps': 12, 'bpp2': 13, 'cuttingstock': 14, 'relaxedClique': 15, 'gap': 16, 'cpmp': 17, 'bpp': 18}

# Load a dictionary where the keys are MILP instance names and the values are the paths to the corresponding lp/mps files
# FIXME: You will need to tailor this process to your application
models = {l.strip().split(",")[0]: l.strip().split(",")[1] for l in open('Names_Paths_and_Labels.csv').readlines()}

# Load a dictionary where the keys are MILP instance names and the values are the corresponding class type
modelLabels = {l.strip().split(",")[0]: labelMap[l.strip().split(",")[2]] for l in open('Names_Paths_and_Labels.csv').readlines()}



# Map from Gurobi's naming convention for variable type into integer values
vtypeAsInt = {'B':1, 'I':-1, 'C':0}

# Map from Gurobi's naming convention for constraint type into integer values
senseMap = {'=':0, '<':-1, '>':1}


# Create a graph for each instance in *models*
for model in models:
    m = read(models[model])
    m._vars = m.getVars()
    m._cons = m.getConstrs()

    numberOfVars = len(m._vars)
    numberOfCons = len(m._cons)
    print('Model: {}   Vars: {}   Cons: {}'.format(model, numberOfVars, numberOfCons))
    nnodes = numberOfVars + numberOfCons
    # NOTE; Nodes 0 - (numberOfVars - 1) will be the variable nodes, Nodes (numberOfVars) - (numberOfVars + numberOfCons - 1) will be the constraint nodes
    # To access the node for constraint j, use (numberOfVars + j)

    # STEP 1: Compile node feature vectors
    #   Construct node feature vectors -- Deep Graph Library will expect a numpy array
    #   The first four node features will be variable-specific, and the last two will be constraint-specific
    #   Feature vector: [var.type, var.lb, var.ub, var.obj, con.rhs, con.sense]
    #   STEP 1.a: Create vectors for variable nodes first...
    firstNode = True
    for i in range(len(m._vars)):
        thisType = vtypeAsInt[m._vars[i].vtype]
        thisLB = m._vars[i].lb
        if thisLB == -0.0:
            thisLB = 0.0
        thisUB = m._vars[i].ub
        # The GCN can't handle *infinity* as a value in its matrix operations ... we need to artificially bound any such such value.
        if math.isinf(thisUB):
            thisUB = 1000
        # Check objective sense before collecting the variable objective coefficient...if objective was Maximization, negate the coefficient
        if m.ModelSense == GRB.MINIMIZE:
            thisOBJ = m._vars[i].obj
        else:
            thisOBJ = -(m._vars[i].obj)
        if firstNode:
            firstNode = False
            # Initialize a 1x6 feature vector as a numpy array
            nodeFeatureVectors = np.array([thisType, thisLB, thisUB, thisOBJ, 0.0, 0.0])
        else:
            # Stack this feature vector underneath the collection of previously compiled feature vectors
            nodeFeatureVectors = np.vstack(( nodeFeatureVectors, np.array([thisType, thisLB, thisUB, thisOBJ, 0.0, 0.0]) ))

    #       Step 1.b: ...then for constraint nodes
    #   NOTE: Since we're already going to be looping through constraints anyway, let's also capture the edges (which connect constraint nodes to active variable nodes) while we're here
    #   When we go to create the edge objects, Deep Graph Library will expect two lists of equal length: *source nodes* and *destination nodes*
    #   It will also (optionally) take a numpy array of edge feature vectors (in this case our edge weights)
    src = []
    dst = []
    indexByVar = {v: i for i, v in enumerate(m._vars)}
    firstEdge = True
    for i in range(len(m._cons)):
        conNodeID = numberOfVars + i
        # get RHS value for this constraint
        thisRHS = m._cons[i].RHS
        # get constraint sense
        thisSense = senseMap[m._cons[i].Sense]
        # add node feature vector for this constraint node to the collection of previously compiled feature vectors
        nodeFeatureVectors = np.vstack(( nodeFeatureVectors, np.array([0.0, 0.0, 0.0, 0.0, thisRHS, thisSense]) ))


        # Step 2: Compile edge information
        # Mark that there will be an edge from this constraint node (conNodeID) to every variable node which appears in this constraint
        # the edge weight will be the variable's coefficient in this constraint
        # access the variables and the coefficients that exist in this constraint
        expr = m.getRow(m._cons[i])
        for j in range(expr.size()):
            varNodeID = indexByVar[expr.getVar(j)]
            thisCoeff = expr.getCoeff(j)
            # add the forward edge
            src.append(conNodeID)
            dst.append(varNodeID)
            # add feature vector for forward edge
            if firstEdge:
                firstEdge = False
                edgeFeatureVectors = np.array([thisCoeff])
            else:
                edgeFeatureVectors = np.vstack((edgeFeatureVectors, np.array([thisCoeff])))
            # NOTE: Deep Graph Library creates directed graphs by default
            # IF we want node feature vectors to be passes across edges in both directions, we have to add each edge in both the forward and backward direction
            # add the backward edge
            src.append(varNodeID)
            dst.append(conNodeID)
            # add identical feature vector for the backward edge
            edgeFeatureVectors = np.vstack(( edgeFeatureVectors, np.array([thisCoeff]) ))

    # Add one self-loop edge for every variable - let the edge attribute be equal to 1.0 so that it send's its info to itself
    # This is to account for isolated variables
    for i in range(nnodes):
        src.append(i)
        dst.append(i)
        edgeFeatureVectors = np.vstack(( edgeFeatureVectors, np.array([1.0]) ))

    # convert src and dst lists to np.array type
    src = np.asarray(src)
    dst = np.asarray(dst)


    # Step 3: Create the actual graph object using Deep Graph Library
    # create the graph object
    g = dgl.DGLGraph()
    # add nodes to the graph
    g.add_nodes(nnodes)
    # add nodes features to the graph (stored in *g.ndata* under a specified name...we'll just call it 'h' ...remember the name you choose, you'll need it to access this data during the ML procedure)
    g.ndata['h'] = torch.as_tensor(nodeFeatureVectors)
    # add edges to the graph
    g.add_edges(src,dst) # Already created edges both ways (i->j and j->i with variable pairs from each constraint, no need to add from src to dst and dst to src here)
    # add edge features to the graph (stored in *g.edata* under a specified name...we'll just call it 'w' ...remember the name you choose, you'll need it to access this data during the ML procedure)
    g.edata['w'] = torch.as_tensor(edgeFeatureVectors)

    # define our graph label (the integer key for this problem type)
    problemType = modelLabels[model]
    # When saving your graph label, DGL expects a dictionary with the specific key 'glabel' and where the value is the integer label stored in a torch tensor
    graph_labels = {'glabel': torch.tensor([problemType])}

    # define a path and save the dgl graph/label object (I'm parsing some info from the path to the lp/mps file and using that to create a new path in a './dgl_graphs' directory)
    # FIXME: You will need to tailor this process to your application
    firstSlash = models[model].find("/")
    lastSlash = models[model].rfind("/")
    save_loc = './dgl_graphs/'+models[model][firstSlash+1:lastSlash+1]
    if not os.path.isdir(save_loc):
        os.makedirs(save_loc)
    # Save the graph: pass this function the (1) path where you'd like to save the binary file, (2) the graph object, and (3) the graph label in the approved DGL format
    save_graphs(save_loc+model+".bin", g, graph_labels)
