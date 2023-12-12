import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class EnhanceEngine:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file, sep="\t", header=None)
        self._preprocess_data()

    def _preprocess_data(self):
        self.df[9] = self.df.iloc[:, 9:].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
        self.df.drop(self.df.iloc[:, 10:], inplace=True, axis=1)
        self.df.columns = ["video ID", "uploader", "age", "category", "length", "views", "rate", "ratings", "comments", "Related IDs"]

    def construct_graph(self):
        G = nx.DiGraph()
        for index, row in self.df.iterrows():
            video_id = row["video ID"]
            related_ids = row["Related IDs"].split(',')

            for related_id in related_ids:
                G.add_edge(video_id, related_id)

        self.graph = G

    def visualize_network(self):
        degrees = dict(self.graph.degree())
        node_sizes = [degrees[node] * 100 for node in self.graph.nodes()]

        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, with_labels=False, node_size=node_sizes, node_color="blue", alpha=0.5, arrows=True)
        plt.title("Comment Network with Node Sizes Reflecting Degrees")
        plt.show()

    def visualize_top_rated_recommendations(self, category, top_n=10):
        category_df = self.df[self.df['category'] == category]
        top_category_videos = category_df.sort_values(by='rate', ascending=False).head(top_n)

        G = nx.DiGraph()

        for index, row in top_category_videos.iterrows():
            video_id = row['video ID']
            related_ids = row['Related IDs'].split(',')

            for related_id in related_ids:
                G.add_edge(video_id, related_id)

        plt.figure(figsize=(12, 12))
        node_colors = ['red' if node in top_category_videos['video ID'].values else 'blue' for node in G.nodes()]
        node_sizes = [700 if node in top_category_videos['video ID'].values else 100 for node in G.nodes()]

        nx.draw(G, with_labels=False, node_size=node_sizes, node_color=node_colors, alpha=0.7, arrows=True)
        plt.title(f"Recommendation Network with Top Rated {category} Videos")
        plt.show()

# Example usage
if __name__ == "__main__":
    engine = EnhanceEngine("Data/0.txt")
    engine.construct_graph()
    engine.visualize_network()

    # Visualize top-rated videos in the 'Comedy' category
    engine.visualize_top_rated_recommendations('Comedy')
