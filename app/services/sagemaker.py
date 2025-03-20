import aiohttp
import json
from typing import List, Dict, Any
from app.core.config import settings

async def predict_standard(data: List[float]) -> Dict[str, Any]:
    """
    Send data to the standard (non-encrypted) SageMaker endpoint.
    
    Args:
        data: List of floating point features
        
    Returns:
        Dictionary with prediction results
    """
    payload = {
        "data": data,
        "encrypted": False,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(settings.SAGEMAKER_STANDARD_ENDPOINT, 
                               json=payload, 
                               headers={"Content-Type": "application/json"}) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"SageMaker standard model error: {error_text}")
                
            result = await response.json()
            return {
                "prediction": float(result["prediction"]),
                "confidence": float(result["confidence"])
            }

async def predict_ckks(encrypted_data: List[str]) -> Dict[str, Any]:
    """
    Send encrypted data to the CKKS-enabled SageMaker endpoint.
    
    Args:
        encrypted_data: List of encrypted features
        
    Returns:
        Dictionary with prediction results
    """
    payload = {
        "data": encrypted_data,
        "encrypted": True,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(settings.SAGEMAKER_CKKS_ENDPOINT, 
                               json=payload, 
                               headers={"Content-Type": "application/json"}) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"SageMaker CKKS model error: {error_text}")
                
            result = await response.json()
            return {
                "prediction": float(result["prediction"]),
                "confidence": float(result["confidence"])
            }
