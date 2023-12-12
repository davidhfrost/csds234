import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    
    def __init__(self, df) -> None:
        self.df = df

        # 9th column onwards are related videos
        self.df[9] = self.df.iloc[:, 9:].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)

        # Convert to an adjacency list of video ids and related video ids
        adj_list = self.df.iloc[:, [0, 9]].set_index(0).to_dict()[9]

        # Separate related video ids into a list
        adj_list = {k: v.split(",") for k, v in adj_list.items()}

        # Create a directed graph
        self.G = nx.DiGraph()

        # Add nodes and edges to the graph
        for source, targets in adj_list.items():
            for target in targets:
                self.G.add_edge(source, target)
    
    def show_graph(self, labels_show = False, node_size = 10)->None:
        # Use the spring_layout algorithm to position nodes
        pos = nx.spring_layout(self.G, seed=42)

        # Draw the graph with adjusted node positions
        plt.figure(figsize=(12, 12))
        nx.draw(self.G, pos, with_labels=labels_show, node_size=node_size )
        plt.title("Directed Graph of related video ids")
        plt.show()


    def show_subgraph(self, root_video_id = 'C0f2dHJ6A18',root_label='Root', labels_show = True, undirected = True, node_size = 100, radius = 3):

        radius = radius
        subgraph = nx.ego_graph(self.G, root_video_id, radius=radius, undirected=undirected)

        # Use the spring_layout algorithm to position nodes
        pos = nx.spring_layout(self.G, seed=42)

        # Calculate the degrees of each node in the subgraph
        node_degrees = dict(subgraph.degree)


        # Your code to generate the subgraph and calculate node degrees

        # Create a colormap based on node degrees
        degree_values = list(node_degrees.values())
        degree_min = min(degree_values)
        degree_max = max(degree_values)
        colormap = plt.cm.get_cmap('coolwarm', degree_max - degree_min + 1)

        # Normalize degrees to the colormap range
        node_colors = [colormap(degree - degree_min) for degree in degree_values]

        # Use the spring_layout algorithm to position nodes
        pos = nx.spring_layout(subgraph, seed=42)

        # Get labels for nodes in the subgraph
        #labels = {node: node for node in subgraph.nodes()}
        labels = {node: root_label if node == root_video_id else node for node in subgraph.nodes()}


        # Draw the subgraph with nodes colored by degree
        plt.figure(figsize=(8, 8))
        nx.draw(
            subgraph, pos, labels=labels, with_labels=labels_show, node_size=node_size,
            node_color=node_colors, cmap=colormap, vmin=degree_min, vmax=degree_max
        )
        plt.title(f"Subgraph centered around {root_video_id} (Radius {radius})")

        # Create a ScalarMappable to associate with the colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=degree_min, vmax=degree_max))
        sm.set_array([])  # You can set an empty array here
        # Add colorbar
        cbar = plt.colorbar(sm, label="Node Degree")
        plt.title(f"Subgraph centered around {root_video_id} (Radius {radius})")
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
        plt.hist(degrees, bins=20, color='skyblue', edgecolor='black')
        plt.title("Degree Distribution", fontsize=16)
        plt.xlabel("Degree", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.show()

    def find_far_away_nvode(self, starting_node='C0f2dHJ6A18'):
        try:
            # Calculate the shortest path lengths from the starting node to all other nodes
            shortest_path_lengths = nx.single_source_shortest_path_length(self.G, source=starting_node)

            # Find the node that is farthest away
            farthest_node = max(shortest_path_lengths, key=shortest_path_lengths.get)
            
            # Print the farthest node and its distance
            print(f"The farthest node from {starting_node} is {farthest_node} with distance {shortest_path_lengths[farthest_node]}")
            
            # Return the farthest node
            return farthest_node
        except nx.NetworkXError:
            print(f"Node {starting_node} not found in the graph.")


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


    def visualize_neighbors(self, node='C0f2dHJ6A18'):
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


    




graph = Graph(df = pd.read_csv("Data/0.txt", sep="\t", header=None))


#graph.show_subgraph()

graph.find_far_away_nodes(graph.G, starting_node='C0f2dHJ6A18', distance_threshold = 10)


graph.visualize_neighbors(node='C0f2dHJ6A18')

graph.visualize_shortest_path()



#graph.show_subgraph()











