from __future__ import annotations

from dataclasses import dataclass


TERMINAL_LIMIT = 256
MAX_REPAIR_INPUT_BYTES = 8192


@dataclass(frozen=True)
class _RepeatedDigram:
    pair: tuple[int, int]
    second_occurrence_order: int


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("value must be non-negative")

    chunks = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            chunks.append(byte | 0x80)
            continue
        chunks.append(byte)
        return bytes(chunks)


def _decode_varint(payload: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if offset >= len(payload):
            raise ValueError("unexpected end of payload while decoding varint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("varint is too large")


def _serialize_binary_grammar(rules: list[tuple[int, int]], start: list[int]) -> bytes:
    output = bytearray()
    output.extend(_encode_varint(len(rules)))
    for left, right in rules:
        output.extend(_encode_varint(left))
        output.extend(_encode_varint(right))
    output.extend(_encode_varint(len(start)))
    for token in start:
        output.extend(_encode_varint(token))
    return bytes(output)


def _deserialize_binary_grammar(payload: bytes) -> tuple[list[tuple[int, int]], list[int]]:
    offset = 0
    rule_count, offset = _decode_varint(payload, offset)

    rules: list[tuple[int, int]] = []
    for _ in range(rule_count):
        left, offset = _decode_varint(payload, offset)
        right, offset = _decode_varint(payload, offset)
        rules.append((left, right))

    start_length, offset = _decode_varint(payload, offset)
    start: list[int] = []
    for _ in range(start_length):
        token, offset = _decode_varint(payload, offset)
        start.append(token)

    if offset != len(payload):
        raise ValueError("unexpected trailing bytes in grammar payload")
    return rules, start


def _expand_tokens(start: list[int], rules: list[tuple[int, int]]) -> bytes:
    expansions: dict[int, bytes] = {}

    def expand(token: int) -> bytes:
        if token < TERMINAL_LIMIT:
            return bytes((token,))
        if token in expansions:
            return expansions[token]

        rule_index = token - TERMINAL_LIMIT
        if rule_index < 0 or rule_index >= len(rules):
            raise ValueError("invalid non-terminal token")
        left, right = rules[rule_index]
        value = expand(left) + expand(right)
        expansions[token] = value
        return value

    output = bytearray()
    for token in start:
        output.extend(expand(token))
    return bytes(output)


def _replace_pair_in_sequence(sequence: list[int], pair: tuple[int, int], symbol: int) -> tuple[list[int], int]:
    replaced: list[int] = []
    replacements = 0
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index] == pair[0] and sequence[index + 1] == pair[1]:
            replaced.append(symbol)
            replacements += 1
            index += 2
            continue
        replaced.append(sequence[index])
        index += 1
    return replaced, replacements


def _count_non_overlapping_occurrences(sequence: list[int], pair: tuple[int, int]) -> tuple[int, int]:
    count = 0
    first_index = -1
    index = 0
    while index + 1 < len(sequence):
        if sequence[index] == pair[0] and sequence[index + 1] == pair[1]:
            if first_index < 0:
                first_index = index
            count += 1
            index += 2
            continue
        index += 1
    return count, first_index


def _remap_active_rules(
    start: list[int],
    rules_by_symbol: dict[int, tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[int]]:
    active_symbols = sorted(rules_by_symbol)
    mapping = {symbol: TERMINAL_LIMIT + index for index, symbol in enumerate(active_symbols)}

    remapped_rules: list[tuple[int, int]] = []
    for symbol in active_symbols:
        left, right = rules_by_symbol[symbol]
        remapped_rules.append((mapping.get(left, left), mapping.get(right, right)))
    remapped_start = [mapping.get(token, token) for token in start]
    return remapped_rules, remapped_start


def _rule_usage(start: list[int], rules: dict[int, list[int]]) -> dict[int, int]:
    usage = {symbol: 0 for symbol in rules}
    for sequence in [start, *rules.values()]:
        for token in sequence:
            if token in usage:
                usage[token] += 1
    return usage


def _inline_rule(start: list[int], rules: dict[int, list[int]], symbol: int) -> list[int]:
    body = list(rules[symbol])

    def inline(sequence: list[int]) -> list[int]:
        expanded: list[int] = []
        for token in sequence:
            if token == symbol:
                expanded.extend(body)
                continue
            expanded.append(token)
        return expanded

    start = inline(start)
    for other_symbol in sorted(rules):
        if other_symbol == symbol:
            continue
        rules[other_symbol] = inline(rules[other_symbol])
    del rules[symbol]
    return start


def _cleanup_rules(start: list[int], rules: dict[int, list[int]]) -> tuple[list[int], bool]:
    changed = False
    while True:
        usage = _rule_usage(start, rules)
        inline_symbol: int | None = None
        for symbol in sorted(rules):
            body = rules[symbol]
            if len(body) < 2 or usage.get(symbol, 0) <= 1:
                inline_symbol = symbol
                break
        if inline_symbol is None:
            return start, changed
        start = _inline_rule(start, rules, inline_symbol)
        changed = True


def encode_repair(text: str) -> bytes:
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    start = list(text.encode("ascii"))
    if len(start) > MAX_REPAIR_INPUT_BYTES:
        return _serialize_binary_grammar([], start)
    rules: list[tuple[int, int]] = []
    next_symbol = TERMINAL_LIMIT

    while len(start) >= 2:
        candidates: dict[tuple[int, int], tuple[int, int]] = {}
        for left, right in zip(start, start[1:]):
            pair = (left, right)
            if pair in candidates:
                continue
            candidates[pair] = _count_non_overlapping_occurrences(start, pair)

        if not candidates:
            break

        pair, (occurrences, first_index) = min(
            (
                (pair, stats)
                for pair, stats in candidates.items()
                if stats[0] >= 3
            ),
            key=lambda item: (-item[1][0], item[1][1], item[0]),
            default=(None, (0, -1)),
        )
        if pair is None or occurrences < 3 or first_index < 0:
            break

        start, replacements = _replace_pair_in_sequence(start, pair, next_symbol)
        if replacements < 3:
            break
        rules.append(pair)
        next_symbol += 1

    return _serialize_binary_grammar(rules, start)


def decode_repair(payload: bytes) -> str:
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes-like")
    rules, start = _deserialize_binary_grammar(bytes(payload))
    return _expand_tokens(start, rules).decode("ascii")


def _find_repeated_digram(start: list[int], rules: dict[int, list[int]]) -> _RepeatedDigram | None:
    seen: dict[tuple[int, int], int] = {}
    order = 0
    sequences = [start, *(rules[symbol] for symbol in sorted(rules))]
    for sequence in sequences:
        for index in range(len(sequence) - 1):
            pair = (sequence[index], sequence[index + 1])
            if pair in seen:
                return _RepeatedDigram(pair=pair, second_occurrence_order=order)
            seen[pair] = order
            order += 1
    return None


def _find_repeated_digram_in_sequence(sequence: list[int]) -> tuple[int, int] | None:
    seen: set[tuple[int, int]] = set()
    repeated_pairs: list[tuple[int, int]] = []
    repeated_seen: set[tuple[int, int]] = set()

    for index in range(len(sequence) - 1):
        pair = (sequence[index], sequence[index + 1])
        if pair in seen:
            if pair not in repeated_seen:
                repeated_pairs.append(pair)
                repeated_seen.add(pair)
            continue
        seen.add(pair)

    for pair in repeated_pairs:
        occurrences, _ = _count_non_overlapping_occurrences(sequence, pair)
        if occurrences >= 2:
            return pair
    return None


def encode_sequitur_style(text: str) -> bytes:
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    start = list(text.encode("ascii"))
    rules: dict[int, tuple[int, int]] = {}
    symbols_by_pair: dict[tuple[int, int], int] = {}
    next_symbol = TERMINAL_LIMIT

    while True:
        pair = _find_repeated_digram_in_sequence(start)
        if pair is None:
            break

        rule_symbol = symbols_by_pair.get(pair)
        if rule_symbol is None:
            rule_symbol = next_symbol
            rules[rule_symbol] = pair
            symbols_by_pair[pair] = rule_symbol
            next_symbol += 1

        start, replacements = _replace_pair_in_sequence(start, pair, rule_symbol)
        if replacements < 2:
            break

    remapped_rules, remapped_start = _remap_active_rules(start, rules)
    return _serialize_binary_grammar(remapped_rules, remapped_start)


def decode_sequitur_style(payload: bytes) -> str:
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("payload must be bytes-like")
    rules, start = _deserialize_binary_grammar(bytes(payload))
    return _expand_tokens(start, rules).decode("ascii")
