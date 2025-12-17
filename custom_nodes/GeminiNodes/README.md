# Gemini API Response Nodes for ComfyUI

This ComfyUI custom node package provides functionality to parse and extract content from Google Gemini API JSON responses.

## Features

- **Gemini Response Parser**: Complete parsing of Gemini API responses, extracting both images and text
- **Gemini Image Extractor**: Extract only images from Gemini responses
- **Gemini Text Extractor**: Extract only text content from Gemini responses
- **Gemini Response Analyzer**: Analyze response structure without extracting content

## Installation

1. Navigate to your ComfyUI `custom_nodes` directory
2. Clone or copy this package:
   ```bash
   git clone <repository-url> GeminiNodes
   ```
3. Restart ComfyUI

## Node Descriptions

### Gemini Response Parser

Main node for parsing Gemini API JSON responses.

**Inputs:**
- `json_data`: JSON string from Gemini API (multiline text)
- `extract_images`: Boolean to enable/disable image extraction
- `extract_text`: Boolean to enable/disable text extraction
- `output_format`: Choose between "tensor" or "pil" format

**Outputs:**
- `images`: Tensor of extracted images (B, C, H, W format)
- `text`: Combined text content from all parts
- `metadata`: Dictionary containing extraction statistics and information

### Gemini Image Extractor

Specialized node for extracting only images from Gemini responses.

**Inputs:**
- `json_data`: JSON string from Gemini API
- `output_format`: Choose between "tensor" or "pil" format

**Outputs:**
- `images`: Tensor of extracted images
- `image_info`: Dictionary with image count and metadata

### Gemini Text Extractor

Specialized node for extracting only text from Gemini responses.

**Inputs:**
- `json_data`: JSON string from Gemini API
- `include_metadata`: Boolean to include usage metadata in output

**Outputs:**
- `text`: Combined text content
- `text_info`: Dictionary with text statistics

### Gemini Response Analyzer

Analyze the structure of Gemini API responses without extracting content.

**Inputs:**
- `json_data`: JSON string from Gemini API
- `detailed_analysis`: Boolean to enable detailed candidate analysis

**Outputs:**
- `analysis`: Dictionary with complete response structure analysis
- `summary`: Human-readable summary string

## Usage Example

1. First, obtain a Gemini API response (you can use API nodes or paste directly)
2. Connect the JSON response to any of the Gemini nodes
3. Extract images/text as needed for your workflow
4. Use the extracted content with other ComfyUI nodes

## Supported Image Formats

- PNG
- JPEG/JPG
- GIF
- WebP
- BMP

## JSON Response Format

The nodes expect Gemini API responses in the standard format:

```json
{
  "modelVersion": "gemini-1.5-pro",
  "responseId": "response-id",
  "usageMetadata": {
    "totalTokenCount": 100,
    "promptTokenCount": 50,
    "candidatesTokenCount": 50
  },
  "candidates": [
    {
      "index": 0,
      "finishReason": "STOP",
      "content": {
        "role": "model",
        "parts": [
          {
            "text": "Some text content"
          },
          {
            "inlineData": {
              "mimeType": "image/png",
              "data": "base64-encoded-image-data"
            }
          }
        ]
      }
    }
  ]
}
```

## Error Handling

The nodes include comprehensive error handling:
- Invalid JSON format detection
- Corrupted image data handling
- Empty response handling
- Detailed error messages in metadata

## Performance Notes

- Images are decoded and converted to tensor format efficiently
- Large responses are processed in memory
- Consider using the specialized nodes (Image/Text Extractor) for better performance when you only need one type of content

## Dependencies

- ComfyUI
- PIL (Pillow) - included with ComfyUI
- torch - included with ComfyUI
- numpy - included with ComfyUI

No additional dependencies required!

## License

This package follows the same license as ComfyUI.

## Contributing

Feel free to submit issues and pull requests to improve these nodes!