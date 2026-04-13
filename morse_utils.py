from __future__ import annotations

from sequence_utils import (
    DEFAULT_BINARY_SYMBOLS,
    SequenceFeatures,
    alternating_identifier,
    extract_sequence_features,
    normalized_sequence_length,
    same_length_runs,
    singleton_runs,
    split_sequence_words,
)


MORSE_CODE: dict[str, str] = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    ".": ".-.-.-",
    ",": "--..--",
    "?": "..--..",
    "'": ".----.",
    "!": "-.-.--",
    "/": "-..-.",
    "(": "-.--.",
    ")": "-.--.-",
    "&": ".-...",
    ":": "---...",
    ";": "-.-.-.",
    "=": "-...-",
    "+": ".-.-.",
    "-": "-....-",
    "_": "..--.-",
    '"': ".-..-.",
    "$": "...-..-",
    "@": ".--.-.",
}
REVERSE_MORSE_CODE = {value: key for key, value in MORSE_CODE.items()}


MorseFeatures = SequenceFeatures


def text_to_morse(text: str, *, letter_sep: str = " ", word_sep: str = " / ") -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    text = text.strip()
    if not text:
        return ""

    encoded_words: list[str] = []
    for word in text.split():
        letters: list[str] = []
        for ch in word:
            code = MORSE_CODE.get(ch.upper())
            if code is None:
                raise ValueError(
                    f"Unsupported character: {ch!r} (U+{ord(ch):04X}). "
                    "Only English letters, digits, and common English punctuation are supported."
                )
            letters.append(code)
        encoded_words.append(letter_sep.join(letters))

    return word_sep.join(encoded_words)


def morse_to_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    text = text.strip()
    if not text:
        return ""

    words: list[str] = []
    for word in text.split("/"):
        word = word.strip()
        if not word:
            continue
        letters: list[str] = []
        for code in word.split():
            letter = REVERSE_MORSE_CODE.get(code)
            if letter is None:
                raise ValueError(f"unsupported Morse code: {code!r}")
            letters.append(letter)
        words.append("".join(letters))
    return " ".join(words)


def split_morse_words(morse: str) -> list[list[str]]:
    return split_sequence_words(morse)


def normalized_morse_length(text: str, morse: str) -> int:
    return normalized_sequence_length(text, morse)


def extract_morse_features(text: str, morse: str) -> MorseFeatures:
    return extract_sequence_features(text, morse, DEFAULT_BINARY_SYMBOLS)
