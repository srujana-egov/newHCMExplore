import os
import json
from typing import Dict, Optional
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import requests
from datetime import datetime

# Configuration
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_CHANNEL = os.getenv('SLACK_REVIEW_CHANNEL', 'rag-review-queue')
RAG_API_URL = "http://localhost:8000/query"  # Your RAG API endpoint

# Initialize Slack client
slack_client = WebClient(token=SLACK_BOT_TOKEN)

def query_rag_api(query: str) -> Dict:
    """Query the RAG API and return the response."""
    try:
        response = requests.post(
            RAG_API_URL,
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def find_similar_nodes(query_embedding: list, k: int = 3) -> list:
    """
    Find k most similar nodes using KNN.
    Returns list of node IDs and their similarity scores.
    """
    # This is a placeholder - implement your KNN logic here
    # You'll need to access your node embeddings and compute similarities
    return [{"node_id": "example_id", "score": 0.95, "label": "Example Node"}]

def send_slack_review_request(query: str, rag_response: str, similar_nodes: list):
    """Send a message to Slack for human review."""
    try:
        # Create blocks for the Slack message
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*New RAG Response Needs Review*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Query:*\n{query}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Proposed Answer:*\n{rag_response}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Similar Nodes (KNN):*"
                }
            }
        ]

        # Add similar nodes
        for node in similar_nodes:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"• {node['label']} (Score: {node['score']:.2f})"
                }
            })

        # Add action buttons
        blocks.extend([
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "✅ Approve"
                        },
                        "style": "primary",
                        "value": json.dumps({
                            "action": "approve",
                            "query": query,
                            "answer": rag_response,
                            "similar_nodes": [n["node_id"] for n in similar_nodes]
                        })
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "✏️ Edit"
                        },
                        "style": "danger",
                        "value": "edit"
                    }
                ]
            }
        ])

        # Send the message
        response = slack_client.chat_postMessage(
            channel=SLACK_CHANNEL,
            blocks=blocks,
            text="New RAG response needs review"
        )
        return response["ts"]  # Return the message timestamp for future reference

    except SlackApiError as e:
        print(f"Error sending message to Slack: {e.response['error']}")
        return None

def handle_slack_interaction(payload: dict):
    """Handle interactions from Slack (button clicks, etc.)"""
    try:
        action = payload["actions"][0]
        action_value = json.loads(action["value"]) if action.get("value") else {}

        if action.get("action_id") == "approve":
            # Handle approval
            add_node_to_graph(
                query=action_value["query"],
                answer=action_value["answer"],
                parent_nodes=action_value["similar_nodes"]
            )
            update_slack_message(
                channel=payload["channel"]["id"],
                ts=payload["message_ts"],
                text="✅ Approved and added to knowledge graph"
            )
        
        elif action.get("action_id") == "edit":
            # Handle edit flow
            update_slack_message(
                channel=payload["channel"]["id"],
                ts=payload["message_ts"],
                text="✏️ Please provide the corrected answer and node position:",
                blocks=create_edit_blocks(payload)
            )

    except Exception as e:
        print(f"Error handling Slack interaction: {e}")

def add_node_to_graph(query: str, answer: str, parent_nodes: list):
    """Add a new node to the knowledge graph."""
    # Generate a new node ID
    new_node_id = f"node_{int(datetime.now().timestamp())}"
    
    # Create the new node
    new_node = {
        "id": new_node_id,
        "label": f"Auto-added: {query[:50]}...",
        "content": answer,
        "created_at": datetime.now().isoformat()
    }
    
    # Add edges to parent nodes
    edges = [{"source": parent_id, "target": new_node_id, "type": "child"} 
            for parent_id in parent_nodes]
    
    # Here you would save the new node and edges to your graph database
    # For example:
    # graph_db.add_node(new_node)
    # for edge in edges:
    #     graph_db.add_edge(edge)
    
    return new_node_id

def create_edit_blocks(payload: dict) -> list:
    """Create blocks for the edit form."""
    original_message = payload["original_message"]
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "✏️ *Edit the answer and position*"
            }
        },
        {
            "type": "input",
            "block_id": "corrected_answer",
            "element": {
                "type": "plain_text_input",
                "multiline": True,
                "action_id": "answer_input"
            },
            "label": {
                "type": "plain_text",
                "text": "Corrected Answer"
            }
        },
        {
            "type": "input",
            "block_id": "parent_nodes",
            "element": {
                "type": "multi_static_select",
                "placeholder": {
                    "type": "plain_text",
                    "text": "Select parent nodes"
                },
                "action_id": "nodes_select",
                "options": [  # You would fetch actual nodes here
                    {
                        "text": {"type": "plain_text", "text": "Node 1"},
                        "value": "node_1"
                    }
                ]
            },
            "label": {
                "type": "plain_text",
                "text": "Parent Nodes"
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Submit Correction"
                    },
                    "style": "primary",
                    "action_id": "submit_correction",
                    "value": json.dumps({
                        "original_ts": payload["message_ts"],
                        "channel_id": payload["channel"]["id"]
                    })
                }
            ]
        }
    ]

def update_slack_message(channel: str, ts: str, text: str, blocks: Optional[list] = None):
    """Update a Slack message."""
    try:
        return slack_client.chat_update(
            channel=channel,
            ts=ts,
            text=text,
            blocks=blocks
        )
    except SlackApiError as e:
        print(f"Error updating Slack message: {e.response['error']}")
