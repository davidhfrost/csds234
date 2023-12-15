import pandas as pd
import sqlite3
import numpy as np
import networkx as nx
import timeit
import matplotlib.pyplot as plt


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
        query = f"SELECT * FROM my_table WHERE {conditions}"
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
        query = f"SELECT * FROM my_table WHERE {conditions}"
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
    
    def approximate_query_processing(self, conditions):
        start_time = timeit.default_timer()
        entity_results = self.entity_search(conditions)
        entity_time = timeit.default_timer() - start_time

        start_time = timeit.default_timer()
        community_results = self.community_search(conditions)
        community_time = timeit.default_timer() - start_time

        num_entity_results = len(entity_results)
        num_community_results = len(community_results)

        # Calculate result similarity (optional, if you need a measure of accuracy)
        similarity_score = self.calculate_similarity(entity_results, community_results)

        print(f"Entity Search - Time: {entity_time} seconds, Results: {num_entity_results}")
        print(f"Community Search - Time: {community_time} seconds, Results: {num_community_results}")
        print(f"Similarity Score: {similarity_score}")

        return {
            'entity_search': {'time': entity_time, 'results': num_entity_results},
            'community_search': {'time': community_time, 'results': num_community_results},
            'similarity_score': similarity_score
        }

    def calculate_similarity(self, results1, results2):
        set1 = set(results1)
        set2 = set(results2)# so i only get unique key 

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        # Calculate Intersection over Union
        if not union:  # Prevent division by zero
            return 0
        iou = len(intersection) / len(union)

        return iou # ovrlappting proportion so i can understand if it belogns to each other overlap 
    
    def plot_approximate_query_results(self, conditions):
        # Run approximate query processing
        approx_results = self.approximate_query_processing(conditions)

        # Extracting the data for plotting
        entity_search_time = approx_results['entity_search']['time']
        community_search_time = approx_results['community_search']['time']
        entity_search_results = approx_results['entity_search']['results']
        community_search_results = approx_results['community_search']['results']
        iou_similarity = approx_results['similarity_score']

        methods = ['Entity Search', 'Community Search']
        times = [entity_search_time, community_search_time]
        results_counts = [entity_search_results, community_search_results]

        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Execution Time (seconds)', color=color)
        ax1.bar(methods, times, color=color, alpha=0.6)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  

        color = 'tab:red'
        ax2.set_ylabel('Number of Results', color=color)  
        ax2.bar(methods, results_counts, color=color, alpha=0.6)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Query Performance and Result Comparison')
        plt.text(1, max(results_counts)/2, f'IoU Similarity: {iou_similarity:.2f}', 
                 horizontalalignment='center', color='green', fontsize=12)

        plt.tight_layout()
        plt.show()
    
    #Test 1: Multi-Condition Query: A query that combines multiple conditions across different columns. 
    def complex_query_1(self):
        conditions = ["category = 'Music'", "views > 5000", "rate > 4.0", "age < 365"]
        query = f"SELECT * FROM my_table WHERE {' AND '.join(conditions)}"
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    #Test 2: Aggregation and Grouping Query: An aggregation query that groups data by a specific column and performs some aggregation function, like counting or averaging, on another column. 
    def complex_query_2(self):
        query = "SELECT category, AVG(length) as average_length, SUM(comments) as total_comments FROM my_table GROUP BY category"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    #Test 3: Multi-Join and Subquery with Aggregation: create a multi-join operation that aggregates data across different dimensions and involves a subquery
    def complex_query_3(self):
        subquery = """
            SELECT v1."video ID" FROM my_table v1
            INNER JOIN my_table v2 ON v1."video ID" = v2."Related IDs"
            WHERE v1.category = v2.category AND v2.views > v1.views
        """
        query = f"""
            SELECT category, AVG(views) as average_views, AVG(rate) as average_rate
            FROM my_table
            WHERE "video ID" IN ({subquery})
            GROUP BY category
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def performance_comparison(self):
        # Define test conditions 1 
        entity_conditions = "category = 'Comedy'"
        ranged_category, min_duration, max_duration = 'Comedy', 100, 500
        community_conditions = "category = 'Comedy'"

        # Measure the execution time for entity_search
        time_entity_search = timeit.timeit(lambda: self.entity_search(entity_conditions), number=100)
        print(f"Entity Search Execution Time: {time_entity_search} seconds")

        # Measure the execution time for ranged_query
        time_ranged_query = timeit.timeit(lambda: self.ranged_query(ranged_category, min_duration, max_duration), number=100)
        print(f"Ranged Query Execution Time: {time_ranged_query} seconds")

        # Measure the execution time for community_search
        time_community_search = timeit.timeit(lambda: self.community_search(community_conditions), number=100)
        print(f"Community Search Execution Time: {time_community_search} seconds")
        
        conditions = "category = 'Comedy'"
        approx_results = engine.approximate_query_processing(conditions)
        print(f" The approximate is :{approx_results}")
        engine.plot_approximate_query_results(conditions)
        
        
        time_complex_query_1 = timeit.timeit(lambda: self.complex_query_1(), number=100)
        time_complex_query_2 = timeit.timeit(lambda: self.complex_query_2(), number=100)
        time_complex_query_3 = timeit.timeit(lambda: self.complex_query_3(), number=100)

        # Plot the performance comparison
        labels = ['Entity Search', 'Ranged Query', 'Community Search', 'Complex Query 1', 'Complex Query 2','Complex Query 3' ]
        execution_times = [time_entity_search, time_ranged_query, time_community_search, time_complex_query_1, time_complex_query_2,time_complex_query_3 ]

        plt.bar(labels, execution_times)
        plt.xlabel('Search Methods')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Performance Comparison of Search Methods')
        plt.show()

# Create an instance of the Engine class
engine = Engine(db="youtube_data.db", df=pd.read_csv("Data/0.txt", sep="\t", header=None))

# Perform the performance comparison for the 'Comedy' category
engine.performance_comparison()








