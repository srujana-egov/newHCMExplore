# search_utils.py
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from data import NODES, EDGES

@dataclass
class SearchResult:
    node_id: str
    score: float
    node_data: dict

class GraphSearcher:
    def __init__(self, nodes: List[dict], edges: List[dict] = None, alpha: float = 0.4, 
                 beta: float = 0.6, t_high: float = 0.78, t_low: float = 0.55):
        self.nodes = nodes
        self.edges = edges or []  # Store edges for parent lookup
        self.alpha = alpha
        self.beta = beta
        self.t_high = t_high
        self.t_low = t_low
        
        # Initialize BM25
        self.tokenized_corpus = [self._tokenize(self._prepare_text(node)) for node in nodes]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Initialize sentence transformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.model.eval()
        
        # Pre-compute embeddings
        print("Computing node embeddings...")
        self.node_texts = [self._prepare_text(node) for node in nodes]
        self.embeddings = self.model.encode(
            self.node_texts, 
            convert_to_tensor=True,
            show_progress_bar=True
        ).cpu().numpy()
        print("Node embeddings computed.")
    
    def _get_parent(self, node: dict) -> Optional[dict]:
        """Find the parent node if it exists."""
        if not hasattr(self, 'edges'):
            return None
            
        for edge in self.edges:
            if edge['target'] == node['id'] and edge.get('type') == 'child':
                return next((n for n in self.nodes if n['id'] == edge['source']), None)
        return None
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text by lowercasing and removing special characters."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return self._normalize_text(text).split()
    
    def _prepare_text(self, node: dict) -> str:
        """Prepare node text with hierarchical context for search."""
        parts = []
        
        # Add parent label if available
        parent = self._get_parent(node)
        if parent:
            parts.append(parent.get('label', '').strip())
        
        # Add current node's label
        label = node.get('label', '').strip()
        parts.append(label)
        
        # Add end_user if available
        end_user = node.get('end_user', '').strip()
        if end_user:
            parts.append(f"(for {end_user})")
        
        # Add a few key terms from content (first 3-4 meaningful words)
        content = node.get('content', '')
        if content:
            key_terms = ' '.join([w for w in content.split() if len(w) > 3][:4])
            if key_terms:
                parts.append(f"- {key_terms}...")
        
        return ' '.join(parts)
    
    def _expand_acronyms(self, text: str) -> str:
        """Expand common acronyms in the text."""
        acronyms = {
            'hcm': 'health campaign management',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience',
            'digit': 'digital infrastructure for governance and transformation',
        }
        words = text.split()
        expanded_words = [acronyms.get(word.lower(), word) for word in words]
        return ' '.join(expanded_words)
    
    def search(self, query: str) -> Tuple[Optional[SearchResult], List[SearchResult]]:
        """
        Search for nodes matching the query.
        Returns (best_match, all_matches_above_threshold)
        """
        # Preprocess query
        query = self._expand_acronyms(query)
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        if bm25_scores.max() > bm25_scores.min():  # Avoid division by zero
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        
        # Get embedding similarity
        with torch.no_grad():  # Disable gradient calculation
            query_embedding = self.model.encode(
                query, 
                convert_to_tensor=True
            ).cpu().numpy()  # Convert to numpy array
            
            # Calculate cosine similarity
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.embeddings
            )[0]
        
        # Combine scores
        combined_scores = (self.alpha * bm25_scores) + (self.beta * similarities)
        
        # Get all results above threshold
        results = []
        for i, score in enumerate(combined_scores):
            if score >= self.t_low:
                results.append(SearchResult(
                    node_id=self.nodes[i]['id'],
                    score=float(score),
                    node_data=self.nodes[i]
                ))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Determine best match
        best_match = results[0] if results and results[0].score >= self.t_high else None
        
        return best_match, results

# Example usage
if __name__ == "__main__":
    
    # Initialize searcher with nodes and edges
    searcher = GraphSearcher(nodes=NODES, edges=EDGES)
    
    # Test search
    test_queries = [
        "what is hcm",
        "how to implement hcm",
        "how do i use hcm",
        "1.8 version"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        best_match, all_matches = searcher.search(query)
        
        if best_match:
            print(f"Best match: {best_match.node_data['label']} (Score: {best_match.score:.2f})")
        elif all_matches:
            print("Did you mean:")
            for i, match in enumerate(all_matches[:3]):  # Show top 3 matches
                print(f"  {i+1}. {match.node_data['label']} (Score: {match.score:.2f})")
        else:
            print("No results found.")
