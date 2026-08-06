from app.core.normalizer import BurmeseTextNormalizer


def test_burmese_text_normalization():
    normalizer = BurmeseTextNormalizer()
    
    # Test zero-width space removal and double space collapse
    raw_input = "ဦးအေးမောင် \u200b  ( အာမခံထားသူ )  "
    normalized = normalizer.normalize(raw_input)
    assert "\u200b" not in normalized
    assert "  " not in normalized
    assert normalized == "ဦးအေးမောင် ( အာမခံထားသူ )"


def test_digit_conversion():
    normalizer = BurmeseTextNormalizer()
    burmese_num = "၁၂၃၄၅၆၇၈၉၀"
    ascii_num = normalizer.normalize_digits_burmese_to_ascii(burmese_num)
    assert ascii_num == "1234567890"
