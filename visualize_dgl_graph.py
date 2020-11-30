import torch as th
from dgl.data.utils import load_graphs


# This is a script to visualize the structure (with no feature vectors) for one specific DGL graph object
# First, use the *load_graphs* function to load a specific graph
#   NOTE: this function returns two values:
#       (1) A python list of graph objects (we are only loading one graph right now, so this list will contain one element)
#       (2) A dictionary containing the label information (in the same format that we saved the label info at the end of the 'create_dgl_graphs.py' script)
glist, label_dict = load_graphs("./dgl_graphs/bppif/ceselli/natural/bpp_nlf_20_9.bin")
g = glist[0]


# Now we are working strictly on a networkx object, and we are going to use matplotlib.pyplot in order to visualize graph structure
import networkx as nx
import matplotlib.pyplot as plt

# Convert our DGL object *g* into a networkx object *nx_G*, and make it an undirected graph (get rid of arrows on edges, since our graphs truly are undirected)
nx_G = g.to_networkx().to_undirected()

# Open a figure in pyplot
fig = plt.figure(dpi=150)
# Clear the figure if it was previously being used by some other script
fig.clf()
# Capture the axes
ax = fig.subplots()

# Generate positions (x,y coordinates) for every edge in our graph.
# ...networkx will do this for you if you specify a layout type (I'm using *kamada_kawai_layout*) and just pass it the entire graph
pos = nx.kamada_kawai_layout(nx_G)

# Draw our graph ... you can toy with the labels, node colors, axes, node sizes, width, etc. as you please
nx.draw_networkx(nx_G, pos, with_labels=False, node_color=[[.3, .3, .3]], ax=ax, node_size=4.0, width=0.5)

# Show our graph structure
plt.show()
