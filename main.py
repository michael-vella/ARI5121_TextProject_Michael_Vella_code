from src.utils.code_evaluation import CodeEvaluation

def format_test_logs(failed_tests: list) -> str:
    new_failures = []
    for failed_test in failed_tests:
        new_failures.append(f"Assertion: {failed_test['test']}. Error: {failed_test['error']}")

    failed_test_cases_str = "\n".join(new_failures)
    return f"### Test Cases where the generated code failed to generate the expected output:\n{failed_test_cases_str}"


code_generated = '''
import hashlib

def string_to_md5(text):
    """
    Given a string 'text', return its md5 hash equivalent string.
    If 'text' is an empty string, return None.

    >>> string_to_md5('Hello world') == '3e25960a79dbc69b674cd4ec67a72c62'
    """
    if not text:
        return None
    return hashlib.md5(text.encode()).hexdigest()
'''


test_cases = """
def check(candidate):

    # Check some simple cases
    assert candidate('Hello world') == '3e25960a79dbc69b674cd4ec67a72c62'
    assert candidate('') == None
    assert candidate('A B C') == '0ef78513b0cb8cef12743f5aeb35f888'
    assert candidate('passwrd') == '5f4dcc3b5aa765d61d8327deb882cf99'

    # Check some edge cases that are easy to work out by hand.
    assert True
"""


f = CodeEvaluation.evaluate_code(code=code_generated, test_cases=test_cases, entry_point="string_to_md5")

print(f)

if not f[0]:
    print(format_test_logs(f[1]))