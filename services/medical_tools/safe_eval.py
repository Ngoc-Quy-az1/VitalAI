from __future__ import annotations

"""Evaluator an toàn cho công thức số học đã được trích xuất từ tài liệu."""

import ast
import operator
from typing import Any


ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

ALLOWED_FUNCTIONS = {
    "min": min,
    "max": max,
}


class FormulaEvaluationError(ValueError):
    """Raised khi expression không an toàn hoặc thiếu biến."""


def safe_eval_expression(expression: str, variables: dict[str, float]) -> float:
    """Evaluate expression số học với AST whitelist.

    Chỉ cho phép `min()`/`max()` hai đối số để hỗ trợ công thức y khoa chuẩn hóa;
    không cho phép attribute access, subscript, import hoặc name lạ.
    """

    tree = ast.parse(expression, mode="eval")
    return float(_eval_node(tree.body, variables))


def _eval_node(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise FormulaEvaluationError("Formula chỉ được chứa hằng số dạng số.")

    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise FormulaEvaluationError(f"Thiếu biến công thức: {node.id}")
        return float(variables[node.id])

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BINOPS:
            raise FormulaEvaluationError(f"Toán tử không được hỗ trợ: {op_type.__name__}")
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        if isinstance(node.op, ast.Pow) and abs(right) > 200:
            raise FormulaEvaluationError("Số mũ quá lớn cho evaluator an toàn.")
        return float(ALLOWED_BINOPS[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARYOPS:
            raise FormulaEvaluationError(f"Toán tử một ngôi không được hỗ trợ: {op_type.__name__}")
        return float(ALLOWED_UNARYOPS[op_type](_eval_node(node.operand, variables)))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise FormulaEvaluationError("Chỉ hỗ trợ hàm min/max trong công thức.")
        if node.keywords or len(node.args) != 2:
            raise FormulaEvaluationError("Hàm min/max phải có đúng 2 đối số.")
        args = [_eval_node(arg, variables) for arg in node.args]
        return float(ALLOWED_FUNCTIONS[node.func.id](*args))

    raise FormulaEvaluationError(f"Expression chứa node không được phép: {type(node).__name__}")


def expression_names(expression: str) -> set[str]:
    """Lấy danh sách biến thực sự xuất hiện trong expression."""

    tree = ast.parse(expression, mode="eval")
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in ALLOWED_FUNCTIONS
    }
