# ==========================================================
# Quality Metrics for Image Evaluation
# CLIP Score: Text-image alignment
# LPIPS: Perceptual similarity to baseline
# ==========================================================

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Lazy imports to avoid loading models unnecessarily
_clip_model = None
_clip_processor = None
_lpips_model = None


# ----------------------------------------------------------
# CLIP Score Computation
# ----------------------------------------------------------

def load_clip_model():
    """Load CLIP model for text-image alignment scoring."""
    global _clip_model, _clip_processor
    
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        
        print("Loading CLIP model...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = _clip_model.to(device)
        _clip_model.eval()
        
        print("CLIP model loaded.")
    
    return _clip_model, _clip_processor


def compute_clip_score(image, prompt):
    """
    Compute CLIP score between image and text prompt.
    
    Args:
        image: PIL Image
        prompt: str
    
    Returns:
        float: CLIP score (higher = better alignment)
    """
    model, processor = load_clip_model()
    device = next(model.parameters()).device
    
    # Process inputs
    inputs = processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Compute similarity
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        score = logits_per_image.item()
    
    return score


# ----------------------------------------------------------
# LPIPS Computation
# ----------------------------------------------------------

def load_lpips_model():
    """Load LPIPS model for perceptual similarity."""
    global _lpips_model
    
    if _lpips_model is None:
        import lpips
        
        print("Loading LPIPS model...")
        _lpips_model = lpips.LPIPS(net='alex')
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _lpips_model = _lpips_model.to(device)
        
        print("LPIPS model loaded.")
    
    return _lpips_model


def pil_to_tensor(image):
    """Convert PIL image to tensor for LPIPS."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    
    # Normalize to [-1, 1]
    img_tensor = img_tensor * 2.0 - 1.0
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


def compute_lpips_score(image1, image2):
    """
    Compute LPIPS perceptual distance between two images.
    
    Args:
        image1: PIL Image (baseline)
        image2: PIL Image (modified method)
    
    Returns:
        float: LPIPS distance (lower = more similar)
    """
    model = load_lpips_model()
    device = next(model.parameters()).device
    
    # Convert images to tensors
    tensor1 = pil_to_tensor(image1).to(device)
    tensor2 = pil_to_tensor(image2).to(device)
    
    # Compute distance
    with torch.no_grad():
        distance = model(tensor1, tensor2)
        score = distance.item()
    
    return score


# ----------------------------------------------------------
# Batch Metrics Computation
# ----------------------------------------------------------

def compute_all_metrics(image, prompt, baseline_image=None):
    """
    Compute all quality metrics for an image.
    
    Args:
        image: PIL Image (generated image)
        prompt: str (text prompt)
        baseline_image: PIL Image (optional, for LPIPS comparison)
    
    Returns:
        dict: {
            'clip_score': float,
            'lpips_score': float or None
        }
    """
    metrics = {}
    
    # CLIP Score
    try:
        metrics['clip_score'] = compute_clip_score(image, prompt)
    except Exception as e:
        print(f"Error computing CLIP score: {e}")
        metrics['clip_score'] = None
    
    # LPIPS Score (only if baseline provided)
    if baseline_image is not None:
        try:
            metrics['lpips_score'] = compute_lpips_score(baseline_image, image)
        except Exception as e:
            print(f"Error computing LPIPS score: {e}")
            metrics['lpips_score'] = None
    else:
        metrics['lpips_score'] = None
    
    return metrics


# ----------------------------------------------------------
# Test
# ----------------------------------------------------------

if __name__ == "__main__":
    
    # Create dummy images for testing
    from PIL import Image
    
    img1 = Image.new('RGB', (512, 512), color='red')
    img2 = Image.new('RGB', (512, 512), color='blue')
    
    prompt = "A red image"
    
    # Test CLIP
    clip_score = compute_clip_score(img1, prompt)
    print(f"CLIP Score: {clip_score:.4f}")
    
    # Test LPIPS
    lpips_score = compute_lpips_score(img1, img2)
    print(f"LPIPS Score: {lpips_score:.4f}")
    
    # Test batch
    metrics = compute_all_metrics(img1, prompt, img2)
    print(f"All Metrics: {metrics}")
