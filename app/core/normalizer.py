import re
import unicodedata


class BurmeseTextNormalizer:
    """Normalizes Burmese (Myanmar) script characters, diacritics, and formatting."""

    # Burmese script Unicode block range: U+1000 to U+109F
    BURMESE_ZERO_WIDTH_SPACE = "\u200b"
    
    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""

        # Apply standard Unicode NFC normalization
        normalized = unicodedata.normalize("NFC", text)

        # Remove zero-width spaces and trailing non-printable control characters
        normalized = normalized.replace(BurmeseTextNormalizer.BURMESE_ZERO_WIDTH_SPACE, "")
        
        # Collapse multiple spaces into single space
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Standardize Burmese digits to English digits if needed, or preserve
        # Standardize common Burmese combining mark diacritic sequences if misplaced
        
        return normalized

    @staticmethod
    def normalize_digits_burmese_to_ascii(text: str) -> str:
        """Converts Burmese numerals (၀-၉) to ASCII digits (0-9)."""
        burmese_digits = "၀၁၂၃၄၅၆၇၈၉"
        ascii_digits = "0123456789"
        trans_table = str.maketrans(burmese_digits, ascii_digits)
        return text.translate(trans_table)
