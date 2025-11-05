import numpy as np
from typing import List, Dict, Any
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

@dataclass
class SimilarNode:
    node_id: str
    score: float
    label: str

class GraphKNN:
    def __init__(self, nodes: List[Dict], embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the KNN model for graph nodes.
        
        Args:
            nodes: List of node dictionaries with at least 'id', 'label', and 'content' keys
            embedding_model_name: Name of the SentenceTransformer model to use
        """
        self.nodes = nodes
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.node_embeddings = None
        self.knn_model = None
        self._prepare_embeddings()
        
    def _prepare_embeddings(self):
        """Generate embeddings for all nodes."""
        # Extract node texts to embed (combine label and content)
        node_texts = [f"{node.get('label', '')} {node.get('content', '')}" 
                     for node in self.nodes]
        
        # Generate embeddings
        self.node_embeddings = self.embedding_model.encode(
            node_texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Initialize KNN model
        self.knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.knn_model.fit(self.node_embeddings)
    
    def find_similar_nodes(self, query_embedding: List[float], k: int = 3) -> List[SimilarNode]:
        """
        Find k most similar nodes to the query embedding.
        
        Args:
            query_embedding: The embedding vector to find similar nodes for
            k: Number of similar nodes to return
            
        Returns:
            List of SimilarNode objects
        """
        if self.knn_model is None:
            raise ValueError("KNN model not initialized. Call _prepare_embeddings() first.")
            
        # Reshape query embedding if needed
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Find k+1 neighbors (in case the query matches a node exactly)
        distances, indices = self.knn_model.kneighbors(query_embedding, n_neighbors=min(k+1, len(self.nodes)))
        
        similar_nodes = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            node = self.nodes[idx]
            # Skip if this is the same node (distance ~= 0)
            if dist < 1e-6:
                continue
                
            similar_nodes.append(SimilarNode(
                node_id=node['id'],
                score=float(1 - dist),  # Convert distance to similarity score
                label=node.get('label', 'Unlabeled Node')
            ))
            
            if len(similar_nodes) >= k:
                break
                
        return similar_nodes
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text query."""
        return self.embedding_model.encode([text])[0].tolist()

# Example usage:
if __name__ == "__main__":
    # Example nodes
    example_nodes = [
        {"id": "1", "label": "Introduction", "content": "This is an introduction to the knowledge graph."},
        {"id": "2", "label": "Installation", "content": "How to install the required packages."},
        # Add more nodes as needed
    ]
    
    # Initialize KNN
    knn = GraphKNN(example_nodes)
    
    # Example query
    query = "How do I set up the environment?"
    query_embedding = knn.get_embedding(query)
    
    # Find similar nodes
    similar = knn.find_similar_nodes(query_embedding, k=2)
    for node in similar:
        print(f"Node: {node.label} (Score: {node.score:.2f})")
