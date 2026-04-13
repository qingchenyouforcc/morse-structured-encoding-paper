from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from codon_utils import codon_words, normalize_codon_sequence


DEFAULT_FIXED_WINDOW_SIZE = 8
MAX_ADAPTIVE_DP_TOKENS = 2048
SEGMENT_SEPARATOR = "#"


@dataclass(frozen=True)
class StructuredCodonEncoderConfig:
    name: str = "structured_codon"
    group_mode: str = "adaptive_window"
    search_mode: str = "base"
    fixed_window_size: int = DEFAULT_FIXED_WINDOW_SIZE
    window_size_choices: tuple[int, ...] = (6, 8, 10, 12)


def encode_structured_codon(sequence: str, config: StructuredCodonEncoderConfig | None = None) -> str:
    """Encode codon tokens with the paper's reference-difference identifier family.

    Codon tokens all have length 3, so a normalized codon sequence is one same-length
    run. The base variant tries one group-level reference-difference segment with
    fallback. The DP variant searches mixtures of raw tokens and legal structured
    subsegments inside that run. `fixed_window` remains available for Exp3 ablation
    by applying the same paper segment encoder within fixed-size chunks.
    """
    config = config or StructuredCodonEncoderConfig()
    tokens = codon_words(normalize_codon_sequence(sequence))
    if not tokens:
        return ""

    if config.group_mode == "fixed_window":
        segments: list[str] = []
        for index in range(0, len(tokens), config.fixed_window_size):
            chunk = tokens[index : index + config.fixed_window_size]
            if config.search_mode == "dp":
                segments.extend(_dp_segments(chunk))
            else:
                segments.append(_best_whole_segment(chunk))
    elif config.search_mode == "dp":
        if len(tokens) > MAX_ADAPTIVE_DP_TOKENS:
            segments = [_best_whole_segment(tokens)]
        else:
            segments = _dp_segments(tokens)
    else:
        segments = [_best_whole_segment(tokens)]

    return SEGMENT_SEPARATOR.join(segments)


def decode_structured_codon(payload: str) -> str:
    if not isinstance(payload, str):
        raise TypeError("payload must be a str")
    body = payload.strip()
    if not body:
        return ""

    tokens: list[str] = []
    for segment in body.split(SEGMENT_SEPARATOR):
        if not segment:
            continue
        if "%" not in segment:
            tokens.extend(_decode_raw_segment(segment))
            continue
        tokens.extend(_decode_structured_segment(segment))
    return " ".join(tokens)


def _dp_segments(tokens: list[str]) -> list[str]:
    n = len(tokens)
    best_costs: list[int | None] = [None] * (n + 1)
    best_segments_at: list[list[str] | None] = [None] * (n + 1)
    best_costs[n] = 0
    best_segments_at[n] = []

    for index in range(n - 1, -1, -1):
        best_cost: int | None = None
        best_segments: list[str] | None = None

        for end in range(index + 1, n + 1):
            raw_segment = _raw_segment(tokens[index:end])
            next_cost = best_costs[end]
            next_segments = best_segments_at[end]
            if next_cost is None or next_segments is None:
                continue
            separator_cost = len(SEGMENT_SEPARATOR) if next_segments else 0
            cost = len(raw_segment) + separator_cost + next_cost
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_segments = [raw_segment, *next_segments]

        for end in range(index + 2, n + 1):
            structured = _structured_segment(tokens[index:end])
            raw_candidate = _raw_segment(tokens[index:end])
            if len(structured) >= len(raw_candidate):
                continue
            next_cost = best_costs[end]
            next_segments = best_segments_at[end]
            if next_cost is None or next_segments is None:
                continue
            separator_cost = len(SEGMENT_SEPARATOR) if next_segments else 0
            cost = len(structured) + separator_cost + next_cost
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_segments = [structured, *next_segments]

        if best_cost is None or best_segments is None:
            raise ValueError("no valid codon DP segmentation found")
        best_costs[index] = best_cost
        best_segments_at[index] = best_segments

    result = best_segments_at[0]
    if result is None:
        raise ValueError("no valid codon DP segmentation found")
    return result


def _best_whole_segment(tokens: list[str]) -> str:
    structured = _structured_segment(tokens)
    raw = _raw_segment(tokens)
    return structured if len(structured) < len(raw) else raw


def _raw_segment(tokens: list[str]) -> str:
    return "".join(tokens)


def _structured_segment(tokens: list[str]) -> str:
    rule = _difference_rule(tokens)
    if rule is None:
        return _raw_segment(tokens)
    source_base, target_base = rule
    identifiers = [_identifier_for_rule(token, source_base, target_base) for token in tokens]
    descriptor = source_base if target_base is None else f"{source_base}{target_base}"
    return "\\".join(identifiers) + f"%{descriptor}"


def _difference_rule(tokens: list[str]) -> tuple[str, str | None] | None:
    """Choose a global diff rule sigma = XY for the segment.

    sigma means: the default base is X, and any position listed in the identifier
    is replaced with Y. A legal structured segment must therefore contain only X/Y
    at every coordinate; tokens outside that 2-symbol explanation space should
    naturally fail fallback because the produced string would not round-trip.
    """
    position_counters = [Counter(token[index] for token in tokens) for index in range(3)]
    default_bases = [min(counter, key=lambda base: (-counter[base], base)) for counter in position_counters]

    pair_scores: dict[tuple[str, str], int] = defaultdict(int)
    alphabet = sorted({base for token in tokens for base in token})
    for source_base in alphabet:
        for target_base in alphabet:
            if source_base == target_base:
                continue
            score = 0
            legal = True
            for token in tokens:
                for base in token:
                    if base not in {source_base, target_base}:
                        legal = False
                        break
                if not legal:
                    break
                score += sum(1 for base in token if base == source_base)
            if legal:
                pair_scores[(source_base, target_base)] = score

    if pair_scores:
        return min(pair_scores, key=lambda pair: (-pair_scores[pair], pair[0], pair[1]))

    if len(alphabet) == 1:
        return alphabet[0], None
    if len(alphabet) < 2:
        return None

    fallback_source = max(set(default_bases), key=default_bases.count)
    fallback_target = next((base for base in alphabet if base != fallback_source), None)
    if fallback_target is None:
        return None
    return fallback_source, fallback_target


def _identifier_for_rule(token: str, source_base: str, target_base: str | None) -> str:
    changes: list[str] = []
    for index, base in enumerate(token, start=1):
        if base == source_base:
            continue
        if target_base is None:
            return token
        if base != target_base:
            return token
        changes.append(str(index))
    return ",".join(changes)


def _decode_structured_segment(segment: str) -> list[str]:
    identifiers_text, descriptor = segment.rsplit("%", 1)
    if len(descriptor) == 1:
        source_base, target_base = descriptor, None
    elif len(descriptor) == 2:
        source_base, target_base = descriptor[0], descriptor[1]
    else:
        raise ValueError(f"invalid codon descriptor: {descriptor!r}")
    identifiers = identifiers_text.split("\\")
    return [_decode_identifier(identifier, source_base, target_base) for identifier in identifiers]


def _decode_identifier(identifier: str, source_base: str, target_base: str | None) -> str:
    if identifier and any(ch not in "123," for ch in identifier):
        return identifier
    token = [source_base, source_base, source_base]
    if identifier == "":
        return "".join(token)
    for change in identifier.split(","):
        if not change.isdigit():
            raise ValueError(f"invalid codon identifier change: {change!r}")
        index = int(change) - 1
        if index < 0 or index >= len(token):
            raise ValueError(f"codon identifier position out of range: {change!r}")
        if target_base is None:
            raise ValueError(f"identifier {change!r} is incompatible with descriptor %{source_base}")
        token[index] = target_base
    return "".join(token)


def _decode_raw_segment(segment: str) -> list[str]:
    if len(segment) % 3 != 0:
        raise ValueError(f"invalid raw codon segment length: {segment!r}")
    tokens = [segment[index : index + 3] for index in range(0, len(segment), 3)]
    return codon_words(normalize_codon_sequence(" ".join(tokens)))
