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


    def show_subgraph(self, root_video_id='C0f2dHJ6A18', root_label='Root', labels_show=True, undirected=True, node_size=100, radius=3):
        subgraph = nx.ego_graph(self.G, root_video_id, radius=radius, undirected=undirected)
        pos = nx.spring_layout(self.G, seed=42)
        node_degrees = dict(subgraph.degree)
        degree_values = list(node_degrees.values())
        degree_min = min(degree_values)
        degree_max = max(degree_values)
        colormap = plt.cm.get_cmap('coolwarm', degree_max - degree_min + 1)
        node_colors = [colormap(degree - degree_min) for degree in degree_values]

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw(subgraph, pos, ax=ax, labels={node: root_label if node == root_video_id else node for node in subgraph.nodes()},
                with_labels=labels_show, node_size=node_size, node_color=node_colors, cmap=colormap, vmin=degree_min, vmax=degree_max)
        plt.title(f"Subgraph centered around {root_video_id} (Radius {radius})")

        # Create a ScalarMappable and associate it with the colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=degree_min, vmax=degree_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Node Degree")  # Pass the Axes object here
        plt.show()

    def find_far_away_nodes(self, starting_node, distance_threshold):
        far_away_nodes = set()
        visited_nodes = set()
        paths = {}  # Store paths to far-away nodes
        queue = [(starting_node, 0, [starting_node])]  # Initialize the queue with the starting node, its distance, and the path

        while queue:
            node, distance, path = queue.pop(0)
            visited_nodes.add(node)

            if distance > distance_threshold:
                far_away_nodes.add(node)
                paths[node] = path  # Store the path

            if distance <= distance_threshold:
                neighbors = list(self.G.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in visited_nodes:
                        queue.append((neighbor, distance + 1, path + [neighbor]))

        return far_away_nodes, paths
    
    def visualize_far_away_nodes(self, starting_node, distance_threshold):
        far_away_nodes, paths = self.find_far_away_nodes(starting_node, distance_threshold)
        print(far_away_nodes)
        subgraph_nodes = set()
        for path in paths.values():
            subgraph_nodes.update(path)
        subgraph = self.G.subgraph(subgraph_nodes)

        pos = nx.spring_layout(subgraph, seed=42)
        plt.figure(figsize=(12, 12))

        # Draw the whole subgraph in a light color
        nx.draw(subgraph, pos, node_color='lightblue', with_labels=True, node_size=50)

        # Highlight the far-away nodes and the paths leading to them
        for node in far_away_nodes:
            path = paths[node]
            nx.draw_networkx_nodes(subgraph, pos, nodelist=path, node_color='red', node_size=100)
            nx.draw_networkx_edges(subgraph, pos, edgelist=nx.edges(subgraph, nbunch=path), edge_color='red', width=2)

        plt.title(f"Nodes Far Away from {starting_node} (Distance Threshold: {distance_threshold})")
        plt.show()
        
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


graph.show_subgraph()

#graph.find_far_away_nodes(starting_node='C0f2dHJ6A18',distance_threshold = 10)

graph.visualize_far_away_nodes(starting_node='C0f2dHJ6A18', distance_threshold=0)
#graph.visualize_neighbors(node='C0f2dHJ6A18')

graph.visualize_shortest_path()



#graph.show_subgraph()