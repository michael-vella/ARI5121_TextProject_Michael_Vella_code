import ast

class CodeEvaluation:
    @staticmethod
    def evaluate_code(code: str, test_cases: str, entry_point: str) -> tuple[bool, list]:
        tests = CodeEvaluation.get_test_list(test_cases)

        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return False, [{"test": ast.unparse(t), "passed": False, "error": f"Code failed to execute: {e}"} for t in tests]

        if entry_point not in namespace:
            return False, [{"test": ast.unparse(t), "passed": False, "error": f"Entry point '{entry_point}' not found"} for t in tests]

        namespace["candidate"] = namespace[entry_point]

        failed = []
        for node in tests:
            test_src = ast.unparse(node)
            try:
                passed = bool(eval(ast.unparse(node.test), namespace))
                if not passed:
                    error = "AssertionError"
                    if isinstance(node.test, ast.Compare):
                        lhs = eval(ast.unparse(node.test.left), namespace)
                        rhs = eval(ast.unparse(node.test.comparators[0]), namespace)
                        error = f"AssertionError: {lhs!r} != {rhs!r}"
                    failed.append({"test": test_src, "passed": False, "error": error})
            except Exception as e:
                failed.append({"test": test_src, "passed": False, "error": f"{type(e).__name__}: {e}"})

        return (True, []) if not failed else (False, failed)

    @staticmethod
    def get_test_list(test_cases: str) -> list[ast.Assert]:
        tree = ast.parse(test_cases)
        return [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]