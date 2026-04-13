from __future__ import annotations


BCD_SYMBOLS: tuple[str, str] = ("0", "1")
BCD_CODE: dict[str, str] = {
    "0": "0000",
    "1": "0001",
    "2": "0010",
    "3": "0011",
    "4": "0100",
    "5": "0101",
    "6": "0110",
    "7": "0111",
    "8": "1000",
    "9": "1001",
}
REVERSE_BCD_CODE = {value: key for key, value in BCD_CODE.items()}


def digits_to_bcd_symbols(text: str, *, code_sep: str = " ", word_sep: str = " / ") -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    text = text.strip()
    if not text:
        return ""

    encoded_words: list[str] = []
    for word in text.split():
        if not word.isdigit():
            raise ValueError(f"unsupported non-digit token: {word!r}")
        encoded_words.append(code_sep.join(BCD_CODE[ch] for ch in word))
    return word_sep.join(encoded_words)


def bcd_symbols_to_digits(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a str")

    sequence = sequence.strip()
    if not sequence:
        return ""

    words: list[str] = []
    for word in sequence.split("/"):
        codes = [code for code in word.strip().split() if code]
        if not codes:
            continue
        digits: list[str] = []
        for code in codes:
            digit = REVERSE_BCD_CODE.get(code)
            if digit is None:
                raise ValueError(f"unsupported BCD code: {code!r}")
            digits.append(digit)
        words.append("".join(digits))
    return " ".join(words)
