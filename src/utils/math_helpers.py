import re


class MathHelpers:
    @staticmethod
    def keep_only_numbers(text: str):
        return re.sub(r'\D', '', str(text))

    @staticmethod
    def get_expected_answer(answer_txt: str):
        idx = answer_txt.find("#### ")
        idx += 5
        return MathHelpers.keep_only_numbers(answer_txt[idx:]) if idx != -1 else answer_txt