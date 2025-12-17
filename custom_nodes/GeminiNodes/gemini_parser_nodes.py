import json
import base64
import torch
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from PIL import Image
import io
import logging

from comfy import model_management
# from comfy.node_base import Node  # Removed - not needed

class GeminiResponseParser:
    """Parse Google Gemini API JSON responses and extract images/text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": ("STRING", {"multiline": True}),
                "extract_images": ("BOOLEAN", {"default": True}),
                "extract_text": ("BOOLEAN", {"default": True}),
                "output_format": (["tensor", "pil"], {"default": "tensor"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "DICT")
    RETURN_NAMES = ("images", "text", "metadata")
    FUNCTION = "parse_response"
    CATEGORY = "API/Gemini"
    DESCRIPTION = "解析Google Gemini API的JSON响应，提取图像和文本内容。支持base64图像解码和全面的元数据提取。当您需要处理包含文本和图像的完整Gemini API响应时使用此节点。"

    def parse_response(self, json_data: str, extract_images: bool,
                      extract_text: bool, output_format: str):
        """
        Parse Gemini API JSON response and extract images and text.

        Args:
            json_data: JSON string from Gemini API response
            extract_images: Whether to extract images
            extract_text: Whether to extract text
            output_format: Output format for images ("tensor" or "pil")

        Returns:
            Tuple of (image_tensor, text_content, metadata_dict)
        """
        try:
            data = json.loads(json_data)

            images = []
            text_parts = []
            metadata = {
                'total_images': 0,
                'total_text_parts': 0,
                'model_version': data.get('modelVersion', 'Unknown'),
                'response_id': data.get('responseId', 'Unknown'),
                'usage_metadata': data.get('usageMetadata', {}),
                'extracted_images': []
            }

            if 'candidates' in data:
                for cand_idx, candidate in enumerate(data['candidates']):
                    if 'content' in candidate and 'parts' in candidate['content']:
                        for part_idx, part in enumerate(candidate['content']['parts']):
                            if 'text' in part and extract_text:
                                text_parts.append(part['text'])
                                metadata['total_text_parts'] += 1

                            elif 'inlineData' in part and extract_images:
                                inline = part['inlineData']
                                mime_type = inline.get('mimeType', 'image/png')
                                b64_data = inline.get('data', '')

                                if b64_data:
                                    try:
                                        decoded_data = base64.b64decode(b64_data)
                                        img_tensor, img_info = self._decode_image(decoded_data, output_format, mime_type)
                                        if img_tensor is not None:
                                            images.append(img_tensor)
                                            metadata['total_images'] += 1
                                            metadata['extracted_images'].append({
                                                'candidate': cand_idx,
                                                'part': part_idx,
                                                'mime_type': mime_type,
                                                'size': len(decoded_data),
                                                'validation': img_info
                                            })
                                    except Exception as e:
                                        logging.warning(f"Error decoding image from candidate {cand_idx}, part {part_idx}: {e}")

            # Convert images to tensor format
            if images:
                image_tensor = torch.cat(images, dim=0)
            else:
                # Return empty tensor with proper dimensions
                image_tensor = torch.zeros((0, 3, 512, 512))

            # Combine text
            combined_text = '\n\n'.join(text_parts) if text_parts else ""

            return (image_tensor, combined_text, metadata)

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {str(e)}"
            return (torch.zeros((0, 3, 512, 512)), error_msg, {'error': error_msg})
        except Exception as e:
            error_msg = f"Error parsing response: {str(e)}"
            return (torch.zeros((0, 3, 512, 512)), error_msg, {'error': error_msg})

    def _decode_image(self, data: bytes, output_format: str, mime_type: str) -> Tuple[Optional[torch.Tensor], Dict]:
        """
        Decode image data to tensor format.

        Args:
            data: Raw image data bytes
            output_format: Output format ("tensor" or "pil")
            mime_type: MIME type of the image

        Returns:
            Tuple of (image_tensor, validation_info)
        """
        validation_info = self._validate_image_data(data, mime_type)

        try:
            img = Image.open(io.BytesIO(data))
            img = img.convert('RGB')

            # Convert to numpy array
            img_array = np.array(img).astype(np.float32) / 255.0

            # Convert to tensor format for ComfyUI (B, C, H, W)
            img_tensor = torch.from_numpy(img_array)
            img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension -> (B, C, H, W)

            return img_tensor, validation_info

        except Exception as e:
            validation_info['status'] = f"Error: {str(e)}"
            return None, validation_info

    def _validate_image_data(self, data: bytes, mime_type: str) -> Dict:
        """Validate image data and extract metadata."""
        import struct

        result = {'status': 'Unknown', 'dimensions': None, 'color_info': None, 'mime_type': mime_type}

        if len(data) < 8:
            result['status'] = 'Invalid: Too small'
            return result

        # Check magic numbers
        magic = data[:8]

        if mime_type == 'image/png':
            if magic.startswith(b'\x89PNG\r\n\x1a\n'):
                try:
                    if len(data) >= 33:
                        ihdr_start = 8
                        ihdr_data = data[ihdr_start+8:ihdr_start+8+13]
                        if len(ihdr_data) == 13:
                            width = struct.unpack('>I', ihdr_data[0:4])[0]
                            height = struct.unpack('>I', ihdr_data[4:8])[0]
                            bit_depth = ihdr_data[8]
                            color_type = ihdr_data[9]

                            result['status'] = 'Valid PNG'
                            result['dimensions'] = (width, height)
                            result['color_info'] = f"Depth: {bit_depth}, Type: {color_type}"
                            return result
                except:
                    pass
                result['status'] = 'Valid PNG (basic)'
            else:
                result['status'] = 'Invalid PNG signature'

        elif mime_type in ['image/jpeg', 'image/jpg']:
            if magic.startswith(b'\xff\xd8\xff'):
                result['status'] = 'Valid JPEG'
            else:
                result['status'] = 'Invalid JPEG signature'

        elif mime_type == 'image/gif':
            if magic.startswith(b'GIF87a') or magic.startswith(b'GIF89a'):
                result['status'] = 'Valid GIF'
            else:
                result['status'] = 'Invalid GIF signature'

        elif mime_type == 'image/webp':
            if magic.startswith(b'RIFF') and data[8:12] == b'WEBP':
                result['status'] = 'Valid WebP'
            else:
                result['status'] = 'Invalid WebP signature'

        elif mime_type == 'image/bmp':
            if magic.startswith(b'BM'):
                result['status'] = 'Valid BMP'
            else:
                result['status'] = 'Invalid BMP signature'

        else:
            result['status'] = f'Unknown MIME type: {mime_type}'

        return result


class GeminiImageExtractor:
    """Extract only images from Gemini API responses."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": ("STRING", {"multiline": True}),
                "output_format": (["tensor", "pil"], {"default": "tensor"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("images", "image_info")
    FUNCTION = "extract_images"
    CATEGORY = "API/Gemini"
    DESCRIPTION = "仅从Gemini API响应中提取图像。针对只需要视觉内容的工作流进行了优化。支持多种图像格式，并提供包括大小和格式信息的详细图像元数据。"

    def extract_images(self, json_data: str, output_format: str):
        """Extract images from Gemini API response."""
        parser = GeminiResponseParser()
        images, _, metadata = parser.parse_response(json_data, True, False, output_format)

        # Create image-specific metadata
        image_info = {
            'count': metadata['total_images'],
            'model': metadata['model_version'],
            'extracted_images': metadata.get('extracted_images', [])
        }

        return (images, image_info)


class GeminiTextExtractor:
    """Extract only text from Gemini API responses."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": ("STRING", {"multiline": True}),
                "include_metadata": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "DICT")
    RETURN_NAMES = ("text", "text_info")
    FUNCTION = "extract_text"
    CATEGORY = "API/Gemini"
    DESCRIPTION = "仅从Gemini API响应中提取文本内容。非常适合不需要图像的文本处理工作流。合并所有文本部分，可选择包含使用元数据，如token计数和响应ID。"

    def extract_text(self, json_data: str, include_metadata: bool):
        """Extract text from Gemini API response."""
        parser = GeminiResponseParser()
        _, text, metadata = parser.parse_response(json_data, False, True, "tensor")

        text_info = {
            'parts_count': metadata['total_text_parts'],
            'model': metadata['model_version']
        }

        if include_metadata:
            text_info['usage_metadata'] = metadata['usage_metadata']
            text_info['response_id'] = metadata['response_id']

        return (text, text_info)


class GeminiResponseAnalyzer:
    """Analyze Gemini API response structure without extracting content."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": ("STRING", {"multiline": True}),
                "detailed_analysis": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("DICT", "STRING")
    RETURN_NAMES = ("analysis", "summary")
    FUNCTION = "analyze_response"
    CATEGORY = "API/Gemini"
    DESCRIPTION = "分析Gemini API响应结构而不提取内容。提供关于token使用情况、候选数量、内容类型和响应有效性的洞察。用于调试API响应或在处理前了解响应结构。"

    def analyze_response(self, json_data: str, detailed_analysis: bool):
        """Analyze Gemini API response structure."""
        try:
            data = json.loads(json_data)

            analysis = {
                'model_version': data.get('modelVersion', 'Unknown'),
                'response_id': data.get('responseId', 'Unknown'),
                'usage_metadata': data.get('usageMetadata', {}),
                'num_candidates': len(data.get('candidates', [])),
                'candidates_info': []
            }

            summary_parts = []
            summary_parts.append(f"Model: {analysis['model_version']}")

            # Usage metadata
            if analysis['usage_metadata']:
                usage = analysis['usage_metadata']
                summary_parts.append(f"Total tokens: {usage.get('totalTokenCount', 'N/A')}")

            # Candidates analysis
            if 'candidates' in data:
                summary_parts.append(f"Candidates: {len(data['candidates'])}")

                for i, candidate in enumerate(data['candidates']):
                    cand_info = {
                        'index': candidate.get('index', i),
                        'finish_reason': candidate.get('finishReason', 'Unknown'),
                        'num_parts': 0,
                        'has_text': False,
                        'has_images': False
                    }

                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        cand_info['num_parts'] = len(parts)

                        for part in parts:
                            if 'text' in part:
                                cand_info['has_text'] = True
                            elif 'inlineData' in part:
                                cand_info['has_images'] = True

                    analysis['candidates_info'].append(cand_info)

                    if detailed_analysis:
                        cand_summary = f"  Candidate {i}: {cand_info['num_parts']} parts"
                        if cand_info['has_text']:
                            cand_summary += " (text)"
                        if cand_info['has_images']:
                            cand_summary += " (images)"
                        summary_parts.append(cand_summary)

            summary = '\n'.join(summary_parts)

            return (analysis, summary)

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {str(e)}"
            return ({'error': error_msg}, error_msg)
        except Exception as e:
            error_msg = f"Error analyzing response: {str(e)}"
            return ({'error': error_msg}, error_msg)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeminiResponseParser": GeminiResponseParser,
    "GeminiImageExtractor": GeminiImageExtractor,
    "GeminiTextExtractor": GeminiTextExtractor,
    "GeminiResponseAnalyzer": GeminiResponseAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiResponseParser": "Gemini Response Parser",
    "GeminiImageExtractor": "Gemini Image Extractor",
    "GeminiTextExtractor": "Gemini Text Extractor",
    "GeminiResponseAnalyzer": "Gemini Response Analyzer",
}

# For backward compatibility
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']