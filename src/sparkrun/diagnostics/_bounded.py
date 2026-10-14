"""Small built-in JSON snapshots: reject before arbitrary traversal or conversion."""

import math


def payload_copy(value):
    nodes = 1024
    text = 8192

    def visit(item, depth):
        nonlocal nodes, text
        nodes -= 1
        if nodes < 0 or depth > 8:
            raise ValueError("payload traversal limit")
        kind = type(item)
        if kind is str:
            if len(item) > 2048:
                raise ValueError("payload string limit")
            text -= len(item)
            if text < 0:
                raise ValueError("payload text limit")
            return item
        if item is None or kind is bool:
            return item
        if kind is int:
            if not -(1 << 63) <= item < (1 << 63):
                raise ValueError("payload integer limit")
            return item
        if kind is float:
            if not math.isfinite(item):
                raise ValueError("payload number must be finite")
            return item
        if kind is dict:
            if len(item) > 64:
                raise ValueError("payload object limit")
            result = {}
            for key, child in item.items():
                if type(key) is not str or len(key) > 128:
                    raise ValueError("payload key limit")
                visit(key, depth + 1)
                result[key] = visit(child, depth + 1)
            return result
        if kind is list:
            if len(item) > 64:
                raise ValueError("payload list limit")
            return [visit(child, depth + 1) for child in item]
        raise ValueError("payload requires built-in JSON values")

    if type(value) is not dict:
        raise ValueError("payload must be an object")
    return visit(value, 0)
