"""
Utility to check available models on the VDL server.

Usage:
    python -c "from backend.app.modules.qa.tests.helpers.vdl_model_checker import list_vdl_models; list_vdl_models()"
"""
import os
from pathlib import Path

import requests
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def list_vdl_models() -> List[Dict[str, Any]]:
    """
    List all available models on the VDL server.
    
    Returns:
        List of model information dictionaries
        
    Raises:
        RuntimeError: If unable to connect to VDL server or API key is missing
    """
    api_key = os.environ.get("VDL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VDL_API_KEY environment variable not set. "
            "Please set it in your .env file or environment."
        )
    
    vdl_base_url = "https://chat.vdl.sdq.kastel.kit.edu/api"
    
    try:
        response = requests.get(
            f"{vdl_base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Handle different API response formats
        models = data.get("data", data)
        if isinstance(models, dict):
            models = [models]
        
        return models
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Failed to connect to VDL server at {vdl_base_url}: {e}"
        ) from e


def print_vdl_models():
    """
    Print all available models on the VDL server in a readable format.
    """
    try:
        models = list_vdl_models()
        
        print("\n" + "="*60)
        print("Available Models on VDL Server")
        print("="*60)
        
        if not models:
            print("No models found.")
            return
        
        for i, model in enumerate(models, 1):
            model_id = model.get("id") or model.get("name") or "Unknown"
            print(f"{i}. {model_id}")
            
            # Print additional info if available
            if isinstance(model, dict):
                if "description" in model:
                    print(f"   Description: {model['description']}")
                if "owned_by" in model:
                    print(f"   Owner: {model['owned_by']}")
        
        print("="*60 + "\n")
        
    except RuntimeError as e:
        print(f"\nError: {e}\n")
        raise


if __name__ == "__main__":
    print_vdl_models()
