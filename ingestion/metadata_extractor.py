"""
Metadata Extractor for agricultural documents.

Extracts and infers metadata such as:
- Crop type (wheat, paddy, etc.)
- Disease/pest information
- Region
- Season
- Language
- Source document
"""

import re
from typing import Dict, List, Optional, Set

from utils.logger import setup_logger

logger = setup_logger(__name__)


class MetadataExtractor:
    """Extract and infer metadata from agricultural documents."""

    def __init__(self):
        """Initialize metadata extractor with domain knowledge."""
        # Crop keywords
        self.crops = {
            "wheat": ["wheat", "गेहूँ", "ਗੇਹੂੰ", "gehun", "gehu"],
            "paddy": ["paddy", "rice", "धान", "ਚਾਵਲ", "dhan", "chawal"],
            "maize": ["maize", "corn", "मकई", "ਮੱਕਾ", "makai"],
            "cotton": ["cotton", "कपास", "ਕਪਾਸ", "kapas"],
        }

        # Disease/pest keywords (English, Hindi, Punjabi)
        self.diseases = {
            "leaf_spot": ["leaf spot", "पत्ती धब्बा", "ਪੱਤਾ ਧੱਬਾ", "patti dhba"],
            "blast": ["blast", "ब्लास्ट", "ਬਲਾਸਟ"],
            "powdery_mildew": ["powdery mildew", "चूर्णी फफूंदी", "ਮੋਹਰਾ ਪ੍ਰਮਾਣ"],
            "root_rot": ["root rot", "जड़ गलन", "ਜੜ ਸੜਨ"],
            "shoot_fly": ["shoot fly", "शूट फ्लाई", "ਸ਼ੂਟ ਫਲਾਈ"],
            "stem_borer": ["stem borer", "तना बेधक", "ਤਨੀ ਬਿਧਕ"],
        }

        # Season keywords
        self.seasons = {
            "kharif": ["kharif", "खरीफ", "ਖਰੀਫ", "monsoon", "rainy"],
            "rabi": ["rabi", "रबी", "ਰਬੀ", "winter"],
            "summer": ["summer", "गर्मी", "ਗਰਮੀ"],
            "spring": ["spring", "वसंत", "ਬਸੰਤ"],
        }

        # Region keywords
        self.regions = {
            "punjab": ["punjab", "पंजाब", "ਪੰਜਾਬ"],
            "haryana": ["haryana", "हरियाणा", "ਹਰਿਆਣਾ"],
            "uttar_pradesh": ["uttar pradesh", "उत्तर प्रदेश", "ਉੱਤਰ ਪ੍ਰਦੇਸ", "up"],
            "madhya_pradesh": ["madhya pradesh", "मध्य प्रदेश", "ਮੱਧਯ ਪ੍ਰਦੇਸ", "mp"],
            "maharashtra": ["maharashtra", "महाराष्ट्र", "ਮਹਾਰਾਸ਼ਟ्ਰ"],
            "karnataka": ["karnataka", "कर्नाटक", "ਕਰਨਾਟਕ"],
        }

        # Language patterns
        self.language_patterns = {
            "hindi": [
                r"[ा-ॿ]",  # Devanagari script
                "हिंदी",
                "hindi",
            ],
            "punjabi": [
                r"[ਅ-ੱ]",  # Gurmukhi script
                "पंजाबी",
                "punjabi",
            ],
            "english": [
                r"[a-zA-Z]{4,}",  # English words
            ],
        }

    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the primary language of the text.

        Args:
            text: Text to analyze

        Returns:
            Language code (en, hi, pa) or None
        """
        if not text:
            return None

        # Sample a portion of the text
        sample = text[:2000]
        lang_scores = {}

        for lang, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, sample, re.IGNORECASE))
                score += matches
            lang_scores[lang] = score

        # Return language with highest score
        if lang_scores and max(lang_scores.values()) > 0:
            return max(lang_scores, key=lang_scores.get)

        return "en"  # Default to English

    def extract_crops(self, text: str) -> Set[str]:
        """
        Extract crop types mentioned in the text.

        Args:
            text: Text to analyze

        Returns:
            Set of crop types found
        """
        crops_found = set()
        text_lower = text.lower()

        for crop, keywords in self.crops.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    crops_found.add(crop)

        return crops_found

    def extract_diseases(self, text: str) -> Set[str]:
        """
        Extract disease/pest types mentioned in the text.

        Args:
            text: Text to analyze

        Returns:
            Set of diseases/pests found
        """
        diseases_found = set()
        text_lower = text.lower()

        for disease, keywords in self.diseases.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    diseases_found.add(disease)

        return diseases_found

    def extract_seasons(self, text: str) -> Set[str]:
        """
        Extract season references from the text.

        Args:
            text: Text to analyze

        Returns:
            Set of seasons mentioned
        """
        seasons_found = set()
        text_lower = text.lower()

        for season, keywords in self.seasons.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    seasons_found.add(season)

        return seasons_found

    def extract_regions(self, text: str) -> Set[str]:
        """
        Extract region references from the text.

        Args:
            text: Text to analyze

        Returns:
            Set of regions mentioned
        """
        regions_found = set()
        text_lower = text.lower()

        for region, keywords in self.regions.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    regions_found.add(region)

        return regions_found

    def infer_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Infer metadata from the filename.

        Args:
            filename: Name of the document file

        Returns:
            Dictionary with inferred metadata
        """
        metadata = {}
        filename_lower = filename.lower()

        # Extract season from filename
        if "kharif" in filename_lower:
            metadata["season"] = "kharif"
        elif "rabi" in filename_lower:
            metadata["season"] = "rabi"

        # Extract year from filename (e.g., 2022, 2023)
        year_match = re.search(r"(20\d{2})", filename)
        if year_match:
            metadata["year"] = year_match.group(1)

        return metadata

    def extract_metadata(self, text: str, filename: str) -> Dict[str, any]:
        """
        Extract all metadata from a document.

        Args:
            text: Document text
            filename: Document filename

        Returns:
            Dictionary with extracted metadata
        """
        logger.info(f"Extracting metadata from {filename}")

        metadata = {
            "filename": filename,
            "language": self.detect_language(text),
            "crops": list(self.extract_crops(text)),
            "diseases": list(self.extract_diseases(text)),
            "seasons": list(self.extract_seasons(text)),
            "regions": list(self.extract_regions(text)),
        }

        # Add filename-based inferences
        filename_metadata = self.infer_from_filename(filename)
        metadata.update(filename_metadata)

        logger.info(f"Extracted metadata: {metadata}")
        return metadata


def extract_metadata_from_document(text: str, filename: str) -> Dict[str, any]:
    """
    Convenience function to extract metadata from a document.

    Args:
        text: Document text
        filename: Document filename

    Returns:
        Metadata dictionary
    """
    extractor = MetadataExtractor()
    return extractor.extract_metadata(text, filename)


if __name__ == "__main__":
    # Example usage
    extractor = MetadataExtractor()

    sample_text = """
    Kharif season wheat cultivation guide for Punjab.
    Methods to control shoot fly and stem borer in wheat fields.
    Best fertilizer recommendations for paddy in Haryana.
    """

    metadata = extractor.extract_metadata(sample_text, "wheat_kharif_2023.pdf")
    print(f"Metadata: {metadata}")
