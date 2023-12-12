import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque 


class Graph:
    
    def __init__(self, df) -> None:
        self.df = df
        self.df[9] = self.df.iloc[:, 9:].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
        adj_list = self.df.iloc[:, [0, 9]].set_index(0).to_dict()[9]
        adj_list = {k: v.split(",") for k, v in adj_list.items()}
        self.G = nx.DiGraph()
        for source, targets in adj_list.items():
            for target in targets:
                self.G.add_edge(source, target)
    
    def show_graph(self, labels_show = False, node_size = 10)->None:
        pos = nx.spring_layout(self.G, seed=42)
        plt.figure(figsize=(12, 12))
        nx.draw(self.G, pos, with_labels=labels_show, node_size=node_size )
        plt.title("Graph of related videos")
        plt.show()

    def show_subgraph(self, root_video_id = 'C0f2dHJ6A18',root_label='Root', labels_show = True, undirected = True, node_size = 100, radius = 3):

        radius = radius
        subgraph = nx.ego_graph(self.G, root_video_id, radius=radius, undirected=undirected)
        pos = nx.spring_layout(self.G, seed=42)
        node_degrees = dict(subgraph.degree)

        # Create a colormap based on node degrees
        degree_values = list(node_degrees.values())
        degree_min = min(degree_values)
        degree_max = max(degree_values)
        colormap = plt.cm.get_cmap('coolwarm', degree_max - degree_min + 1)
        # Normalize 
        node_colors = [colormap(degree - degree_min) for degree in degree_values]

        #spring_layout to position
        pos = nx.spring_layout(subgraph, seed=42)

        labels = {node: root_label if node == root_video_id else node for node in subgraph.nodes()}
        #subgraph, nodes intensity by degree
        plt.figure(figsize=(8, 8))
        nx.draw(
            subgraph, pos, labels=labels, with_labels=labels_show, node_size=node_size,
            node_color=node_colors, cmap=colormap, vmin=degree_min, vmax=degree_max
        )
        plt.title(f"Subgraph centered at {root_video_id} (Radius {radius})")
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=degree_min, vmax=degree_max))
        sm.set_array([])  
        cbar = plt.colorbar(sm, label="Node Degree")
        plt.title(f"Subgraph centered at {root_video_id} (Radius {radius})")
        plt.show()

    def find_far_away_nodes(graph, starting_node, distance_threshold):
        far_away_nodes = set()
        visited_nodes = set()
        queue = [(starting_node, 0)]  # Initialize the queue with the starting node and its distance

        while queue:
            node, distance = queue.pop(0)
            visited_nodes.add(node)

            if distance > distance_threshold:
                far_away_nodes.add(node)

            if distance <= distance_threshold:
                neighbors = graph.neighbors(node)
                for neighbor in neighbors:
                    if neighbor not in visited_nodes:
                        queue.append((neighbor, distance + 1))
        return far_away_nodes
    
    def degree_distribution(self):
        degrees = [d for n, d in self.G.degree()]
        plt.figure(figsize=(8, 6))
        plt.hist(degrees, bins=20, color= 'purple', edgecolor='black')
        plt.title("Degree Distribution", fontsize=16)
        plt.xlabel("Degree", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.show()

    def find_far_away_nodes(self, starting_node, distance_threshold):
        far_away_nodes = set()
        visited_nodes = set()
        queue = deque([(starting_node, 0)])  # Using deque for efficient pop from the left

        while queue:
            node, distance = queue.popleft()
            if node not in visited_nodes:
                visited_nodes.add(node)

                if distance >= distance_threshold:
                    far_away_nodes.add(node)

                for neighbor in self.G.neighbors(node):
                    if neighbor not in visited_nodes:
                        queue.append((neighbor, distance + 1))

        return far_away_nodes


    def visualize_shortest_path(self, source = 'C0f2dHJ6A18', target ='BwEakCm3c1c'):
        try:
            path = nx.shortest_path(self.G, source=source, target=target)
            subgraph = self.G.subgraph(path)
            pos = nx.spring_layout(subgraph, seed=42)
            plt.figure(figsize=(8, 8))
            nx.draw(subgraph, pos, with_labels=True, node_size=100)
            plt.title(f"Shortest Path from {source} to {target}")
            plt.show()
        except nx.NetworkXNoPath:
            print(f"No path found from {source} to {target}.")

    def visualize_neighborhood(self, node='C0f2dHJ6A18'):
            try:
                neighbors = list(self.G.neighbors(node))
                if not neighbors:
                    print(f"No neighbors found for node {node}.")
                    return
                subgraph = self.G.subgraph([node] + neighbors)
                pos = nx.spring_layout(subgraph, seed=42)
                plt.figure(figsize=(8, 8))
                nx.draw(subgraph, pos, with_labels=True, node_size=100)
                plt.title(f"Neighbors of Node {node}")
                plt.show()
            except nx.NetworkXError:
                print(f"Node {node} not found in the graph.")

    #BFS
    def find_farthest_node(self, starting_node = 'F9qvkMq7EIE'):
        # Ensure the starting node exists in the graph
        if starting_node not in self.G:
            print(f"Node {starting_node} not found in the graph.")
            return None
        shortest_path_lengths = {}
        queue = deque([(starting_node, 0)]) 
        while queue:
            current_node, distance = queue.popleft()
            shortest_path_lengths[current_node] = distance
            for neighbor in self.G.neighbors(current_node):
                if neighbor not in shortest_path_lengths:
                    queue.append((neighbor, distance + 1))

        # Find the farthest node and its distance
        farthest_node, max_distance = max(shortest_path_lengths.items(), key=lambda x: x[1])

        print(f"The farthest node from {starting_node} is {farthest_node} with distance {max_distance}")
        return farthest_node
    
    def find_globally_longest_path(self):
        max_length = 0
        farthest_pair = (None, None)

        for starting_node in self.G.nodes():
            shortest_path_lengths = nx.single_source_shortest_path_length(self.G, starting_node)
            farthest_node, distance = max(shortest_path_lengths.items(), key=lambda x: x[1])
            if distance > max_length:
                max_length = distance
                farthest_pair = (starting_node, farthest_node)

        print(f"Longest path in graph is from {farthest_pair[0]} to {farthest_pair[1]} with length {max_length}")
        return farthest_pair


    
#('F9qvkMq7EIE', '91_VUWLpx8Y')

#graph = Graph(df = pd.read_csv("Data/0.txt", sep="\t", header=None))


#graph.show_subgraph()

#graph.find_far_away_nodes(starting_node='34RGjnTIi7w', distance_threshold=1)

#graph.find_globally_longest_path()

#graph.find_farthest_node(starting_node)

#graph.visualize_neighborhood(node='C0f2dHJ6A18')

#graph.visualize_shortest_path()

#'34RGjnTIi7w'

#graph.show_subgraph()











