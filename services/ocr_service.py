"""
OCR service using Tesseract
Optimized for:
- English
- Vietnamese
- Code screenshots
- Multilingual documents
"""

import io
import logging
import os
from typing import List
import numpy as np
import cv2
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


class OCRService:
    """
    OCR Service using Tesseract OCR
    """

    def __init__(self):
        """
        Initialize Tesseract OCR
        """
        try:
            # Test if tesseract is installed
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR initialized: v{version}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            logger.error("Please install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
            raise

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def extract_text(
        self,
        image_content: bytes,
        enhance: bool = True,
        language: str = "auto"
    ) -> str:
        """
        Extract text from image bytes

        Args:
            image_content: image bytes
            enhance: apply image enhancement for OCR
            language: 'en', 'vi', 'code', 'auto' (Tesseract auto-detects)

        Returns:
            extracted text
        """

        try:
            img = self._load_image(image_content)
            
            # Determine Tesseract language config
            if language == "vi":
                lang_config = "vie"
            elif language in ("en", "code"):
                lang_config = "eng"
            else:  # auto
                lang_config = "eng+vie"  # Try both
            
            # Try OCR without enhancement first
            text = self._ocr_image(img, lang_config)
            
            # If result is poor and enhancement is enabled, try with enhancement
            if enhance and self._is_poor_result(text):
                logger.info("Poor OCR result, retrying with enhancement")
                img_enhanced = self._enhance_for_ocr(img)
                text_enhanced = self._ocr_image(img_enhanced, lang_config)
                
                # Use better result
                if len(text_enhanced) > len(text):
                    text = text_enhanced
            
            logger.info(f"OCR extracted {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}", exc_info=True)
            return ""

    # ==========================================================
    # INTERNAL HELPERS
    # ==========================================================
    
    def _ocr_image(self, img: np.ndarray, lang: str) -> str:
        """
        Perform OCR on image using Tesseract
        """
        try:
            # Convert numpy array back to PIL Image for pytesseract
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            
            # Tesseract config for better accuracy
            custom_config = r'--oem 3 --psm 6'
            
            # Perform OCR
            text = pytesseract.image_to_string(
                pil_image,
                lang=lang,
                config=custom_config
            )
            
            # Clean up
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""

    def _load_image(self, image_content: bytes) -> np.ndarray:
        """
        Load image bytes into OpenCV BGR format
        """
        try:
            image = Image.open(io.BytesIO(image_content))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            logger.debug(f"Image loaded: shape={img_array.shape}")
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def _enhance_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for better OCR
        Lighter touch to preserve text structure
        """

        try:
            # Upscale if image is small (improves OCR accuracy)
            height, width = img.shape[:2]
            if height < 1000 or width < 1000:
                scale_factor = 2.0
                img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, 
                               interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Light denoising only
            denoised = cv2.fastNlMeansDenoising(gray, h=10)

            # Gentle contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # Simple thresholding - less aggressive
            _, binary = cv2.threshold(enhanced, 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Back to BGR for PaddleOCR
            result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            return result

        except Exception as e:
            logger.warning(f"Image enhancement failed, using original: {e}")
            return img
    
    def _is_poor_result(self, text: str) -> bool:
        """
        Check if OCR result is poor quality
        """
        if not text or len(text.strip()) < 10:
            return True
        
        # Check for too many single characters (like "n a o t o e e")
        words = text.split()
        if not words:
            return True
        single_chars = sum(1 for w in words if len(w) == 1)
        if single_chars / len(words) > 0.5:
            return True
        
        return False