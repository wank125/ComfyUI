import json
import torch
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import io
import logging
import time

from comfy import model_management
# from comfy.node_base import Node  # Removed - not needed
from comfy.utils import ProgressBar


class GeminiImageGenerator:
    """
    Generate images using Gemini 2.5 Flash Image Preview API
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "image_size": (["1K", "2K"], {"default": "2K"}),
                "candidate_count": ("INT", {"default": 1, "min": 1, "max": 4}),
                "timeout": ("INT", {"default": 60, "min": 10, "max": 300}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31-1}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "base_url": ("STRING", {"default": "https://xiaoai.plus/v1beta/models"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "DICT")
    RETURN_NAMES = ("image", "metadata", "api_response")
    FUNCTION = "generate"
    CATEGORY = "API/Gemini"
    DESCRIPTION = "使用Google Gemini 2.5 Flash图像预览API生成图像。支持文本到图像生成，具有可自定义的宽高比、尺寸、负面提示和种子以确保结果可重现。返回生成的图像以及详细的API响应元数据。"

    OUTPUT_NODE = False

    def generate(self, api_key: str, prompt: str, aspect_ratio: str, image_size: str,
                 candidate_count: int, timeout: int, seed: int = -1,
                 negative_prompt: str = "", base_url: str = "https://xiaoai.plus/v1beta/models") -> Tuple[torch.Tensor, str, Dict]:
        """
        Generate image using Gemini API

        Returns:
            Tuple of (image_tensor, text_content, metadata_dict)
        """
        if not api_key:
            raise ValueError("API key is required")

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Prepare request
        url = f"{base_url}/gemini-2.5-flash-image-preview:generateContent?key={api_key}"

        headers = {
            "Content-Type": "application/json"
        }

        # Build request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "candidateCount": candidate_count,
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size
                }
            }
        }

        # Add seed if provided
        if seed != -1:
            payload["generationConfig"]["seed"] = seed

        # Add negative prompt if provided
        if negative_prompt.strip():
            # Add as second part in content
            payload["contents"][0]["parts"].append({
                "text": f"Negative prompt: {negative_prompt}"
            })

        # Make API call
        try:
            response_data = self._make_api_call(url, headers, payload, timeout)

            # Parse response using existing parser
            from .gemini_parser_nodes import GeminiResponseParser
            parser = GeminiResponseParser()

            # Convert response to JSON string for parser
            json_str = json.dumps(response_data)

            # Use parser to extract image and metadata
            images, _, metadata = parser.parse_response(json_str, True, False, "tensor")

            # Generate text content from response
            text_content = f"Generated image using Gemini 2.5 Flash\n"
            text_content += f"Model: gemini-2.5-flash-image-preview\n"
            text_content += f"Aspect ratio: {aspect_ratio}\n"
            text_content += f"Size: {image_size}\n"

            if metadata.get('usage_metadata'):
                usage = metadata['usage_metadata']
                text_content += f"\nUsage:\n"
                text_content += f"  Total tokens: {usage.get('totalTokenCount', 'N/A')}\n"
                text_content += f"  Prompt tokens: {usage.get('promptTokenCount', 'N/A')}\n"
                text_content += f"  Candidates tokens: {usage.get('candidatesTokenCount', 'N/A')}\n"

            # Add original API response to metadata
            metadata['api_request'] = {
                'prompt': prompt,
                'aspect_ratio': aspect_ratio,
                'image_size': image_size,
                'candidate_count': candidate_count,
                'seed': seed if seed != -1 else None,
                'negative_prompt': negative_prompt if negative_prompt.strip() else None
            }

            # Also return the raw JSON response for parsing nodes
            api_response_json = json_str

            return images[0] if len(images) > 0 else torch.zeros((1, 3, 512, 512)), text_content, api_response_json

        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            logging.error(error_msg)
            return torch.zeros((1, 3, 512, 512)), error_msg, json.dumps({"error": error_msg})

    def _make_api_call(self, url: str, headers: Dict, payload: Dict, timeout: int) -> Dict:
        """Make synchronous API call"""
        import requests

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            raise Exception(f"Request timed out after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise Exception("Failed to connect to API. Check your internet connection.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise Exception("Invalid API key. Please check your credentials.")
            elif e.response.status_code == 429:
                raise Exception("Rate limit exceeded. Please try again later.")
            elif e.response.status_code == 400:
                raise Exception("Bad request. Check your input parameters.")
            else:
                raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from API")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")


class GeminiTextToImageAdvanced:
    """
    Advanced text-to-image generation with more options
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["gemini-2.5-flash-image-preview", "gemini-2.0-flash-exp"], {"default": "gemini-2.5-flash-image-preview"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "image_size": (["1K", "2K"], {"default": "2K"}),
                "quality": (["standard", "high"], {"default": "standard"}),
                "style": (["natural", "vivid", "dramatic"], {"default": "natural"}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31-1}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "base_url": ("STRING", {"default": "https://xiaoai.plus/v1beta/models"}),
                "custom_config": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "DICT")
    RETURN_NAMES = ("image", "metadata", "full_response")
    FUNCTION = "generate_advanced"
    CATEGORY = "API/Gemini"
    DESCRIPTION = "高级Gemini文本到图像生成，具有增强的控制功能。提供多种模型、质量设置、样式预设和自定义配置选项。非常适合需要对图像生成参数和输出特征进行精确控制的专业工作流。"

    def generate_advanced(self, api_key: str, prompt: str, model: str, aspect_ratio: str,
                          image_size: str, quality: str, style: str, seed: int = -1,
                          negative_prompt: str = "", base_url: str = "https://xiaoai.plus/v1beta/models",
                          custom_config: str = "") -> Tuple[torch.Tensor, str, Dict]:
        """
        Advanced image generation with more options
        """
        # Build URL
        url = f"{base_url}/{model}:generateContent?key={api_key}"

        headers = {"Content-Type": "application/json"}

        # Build content
        content_parts = [{"text": prompt}]

        if negative_prompt.strip():
            content_parts.append({
                "text": f"Avoid: {negative_prompt}"
            })

        # Base payload
        payload = {
            "contents": [{"parts": content_parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "candidateCount": 1,
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                    "quality": quality,
                    "style": style
                }
            }
        }

        # Add seed if provided
        if seed != -1:
            payload["generationConfig"]["seed"] = seed

        # Parse and merge custom config if provided
        if custom_config.strip():
            try:
                custom_json = json.loads(custom_config)
                # Deep merge
                self._deep_merge(payload, custom_json)
            except json.JSONDecodeError:
                logging.warning("Invalid custom_config JSON, ignoring")

        # Make API call using the same method as basic generator
        generator = GeminiImageGenerator()
        response_data = generator._make_api_call(url, headers, payload, 120)

        # Parse response
        from .gemini_parser_nodes import GeminiResponseParser
        parser = GeminiResponseParser()
        json_str = json.dumps(response_data)
        images, _, metadata = parser.parse_response(json_str, True, False, "tensor")

        # Enhanced metadata
        text_metadata = f"Generated with {model}\n"
        text_metadata += f"Style: {style}, Quality: {quality}\n"
        text_metadata += f"Config: {aspect_ratio}, {image_size}\n"

        if metadata.get('usage_metadata'):
            usage = metadata['usage_metadata']
            text_metadata += f"\nTokens: {usage.get('totalTokenCount', 'N/A')} total\n"

        return images[0] if len(images) > 0 else torch.zeros((1, 3, 512, 512)), text_metadata, json.dumps(response_data)

    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


class GeminiAPITester:
    """
    Test Gemini API connectivity and get model info
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "test_type": (["ping", "models", "quota"], {"default": "ping"}),
                "base_url": ("STRING", {"default": "https://xiaoai.plus/v1beta"}),
            }
        }

    RETURN_TYPES = ("STRING", "DICT")
    RETURN_NAMES = ("result", "info")
    FUNCTION = "test_api"
    CATEGORY = "API/Gemini"
    DESCRIPTION = "测试Gemini API连接并检索服务信息。支持ping测试、模型可用性检查和配额状态查询。是验证API密钥有效性和生成任务前服务可用性的重要故障排除工具。"

    def test_api(self, api_key: str, test_type: str, base_url: str) -> Tuple[str, Dict]:
        """Test API connection and get information"""
        import requests

        if not api_key:
            return "Error: API key is required", {"error": "missing_api_key"}

        results = {"test_type": test_type, "timestamp": time.time()}

        try:
            if test_type == "ping":
                # Simple ping by generating a tiny image
                url = f"{base_url}/models/gemini-2.5-flash-image-preview:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": "test"}]}],
                    "generationConfig": {
                        "responseModalities": ["IMAGE"],
                        "imageConfig": {"imageSize": "1K"}
                    }
                }

                response = requests.post(url, json=payload, timeout=10)

                if response.status_code == 200:
                    results["status"] = "success"
                    results["response_time"] = response.elapsed.total_seconds()
                    return f"✅ API is accessible! Response time: {results['response_time']:.2f}s", results
                else:
                    results["status"] = "error"
                    results["http_code"] = response.status_code
                    return f"❌ API error: HTTP {response.status_code}", results

            elif test_type == "models":
                # List available models
                url = f"{base_url}/models?key={api_key}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    models = response.json().get("models", [])
                    results["models"] = models
                    model_list = "\n".join([f"- {m.get('name', 'Unknown')}" for m in models])
                    return f"✅ Available models:\n{model_list}", results
                else:
                    return f"❌ Failed to list models: HTTP {response.status_code}", results

            elif test_type == "quota":
                # Try to get quota info (may not be available on all endpoints)
                url = f"{base_url}/models/gemini-2.5-flash-image-preview:generateContent?key={api_key}"
                payload = {
                    "contents": [{"parts": [{"text": "quota test"}]}],
                    "generationConfig": {
                        "responseModalities": ["TEXT"],
                        "maxOutputTokens": 1
                    }
                }

                response = requests.post(url, json=payload, timeout=10)

                if response.status_code == 200:
                    resp_data = response.json()
                    if 'usageMetadata' in resp_data:
                        results["usage"] = resp_data['usageMetadata']
                        usage = resp_data['usageMetadata']
                        return f"✅ Quota info:\n{json.dumps(usage, indent=2)}", results
                    else:
                        return "✅ API accessible, but quota info not available", results
                else:
                    return f"❌ Failed to check quota: HTTP {response.status_code}", results

        except Exception as e:
            results["error"] = str(e)
            return f"❌ Test failed: {str(e)}", results


# Register nodes
NODE_CLASS_MAPPINGS = {
    "GeminiImageGenerator": GeminiImageGenerator,
    "GeminiTextToImageAdvanced": GeminiTextToImageAdvanced,
    "GeminiAPITester": GeminiAPITester,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageGenerator": "Gemini Image Generator",
    "GeminiTextToImageAdvanced": "Gemini Advanced Image Gen",
    "GeminiAPITester": "Gemini API Tester",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']