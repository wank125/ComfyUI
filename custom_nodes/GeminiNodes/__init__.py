"""
ComfyUI Custom Nodes for Gemini API

This package provides nodes to interact with Google Gemini API for image generation,
response parsing, and API testing.
"""

from .gemini_parser_nodes import NODE_CLASS_MAPPINGS as PARSER_NODES, NODE_DISPLAY_NAME_MAPPINGS as PARSER_DISPLAY_NAMES
from .gemini_api_nodes import NODE_CLASS_MAPPINGS as API_NODES, NODE_DISPLAY_NAME_MAPPINGS as API_DISPLAY_NAMES

# Merge all node mappings
NODE_CLASS_MAPPINGS = {**PARSER_NODES, **API_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**PARSER_DISPLAY_NAMES, **API_DISPLAY_NAMES}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']