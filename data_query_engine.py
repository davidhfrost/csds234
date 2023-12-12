import pandas as pd
import sqlite3
import numpy as np
import networkx as nx


class Engine:

    def __init__(self, db, df) -> None:
        self.db = db
        self.df = df


        self.conn = sqlite3.connect(self.db)
        self.df[9] = self.df.iloc[:, 9:].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
        adj_list = self.df.iloc[:, [0, 9]].set_index(0).to_dict()[9]
        adj_list = {k: v.split(",") for k, v in adj_list.items()}
        self.graph = nx.DiGraph()
        for source, targets in adj_list.items():
            for target in targets:
                self.graph.add_edge(source, target)

        self.df.drop(self.df.iloc[:, 10:], inplace=True, axis=1)
        self.df.columns = ["video ID", "uploader", "age", "category", "length", "views", "rate", "ratings", "comments", "Related IDs"]
        self.df.to_sql('my_table', self.conn, if_exists='replace', index=False)
        self.cursor =self.conn.cursor()

    def entity_search(self, conditions):
        query = f"SELECT * FROM my_table WHERE {' AND '.join(conditions)}"
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def output_entity_search(self, conditions):
        results = engine.entity_search([conditions] )
        for result in results:
            print(result)

    
    def ranged_query(self, category, min_duration, max_duration):
        query = f"SELECT * FROM my_table WHERE category = ? AND length BETWEEN ? AND ?"
        self.cursor.execute(query, (category, min_duration, max_duration))
        return self.cursor.fetchall()
    
    def output_ranged_query(self, category, min_duration, max_duration):
        results = engine.ranged_query(category, min_duration, max_duration)
        for result in results:
            print(result)

    def community_search(self, conditions):
        category_condition = conditions[0]
        query = f"SELECT * FROM my_table WHERE {category_condition}"
        self.cursor.execute(query)
        community_results = self.cursor.fetchall()
        return community_results
    
    def output_communiy_search(self, conditions):
        results = engine.community_search([conditions])
        for result in results:
            print(result)


    def construct_graph(self):
        # 9th column onwards are related videos
        adj_list = self.df.iloc[:, [0, 9]].set_index(0).to_dict()[9]
        adj_list = {k: v.split(",") for k, v in adj_list.items()}
        self.graph = nx.DiGraph()
        for source, targets in adj_list.items():
            for target in targets:
                self.graph.add_edge(source, target)

    
    def aggregate_nodes_by_type(self, type = "category"):
        # Create a new graph for aggregation
        aggregated_graph = nx.DiGraph()

        # Group nodes by the  attribute and aggregate
        for attribute in set(nx.get_node_attributes(self.graph, type).values()):
            nodes_in_attribute = [node for node, data in self.graph.nodes(data=True) if data.get(type) == attribute]
            if nodes_in_attribute:
                aggregated_graph.add_node(attribute)
                for u in nodes_in_attribute:
                    for v in nodes_in_attribute:
                        if u != v and self.graph.has_edge(u, v):
                            aggregated_graph.add_edge(attribute, v)

        return aggregated_graph













engine = Engine(db="youtube_data.db", df=pd.read_csv("Data/0.txt", sep="\t", header=None))
#engine.output_ranged_query('Comedy', 60, 120)

engine.output_entity_search("category = 'Comedy'")




engine.aggregate_nodes_by_type()



