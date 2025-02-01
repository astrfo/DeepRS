import re

def trimming_trailing_zero(value):
    if isinstance(value, (int, float)):
        value_str = str(value)
        value_str = re.sub(r'(\.\d*?[1-9])0+$', r'\1', value_str)  # 末尾の 0 を削除
        value_str = re.sub(r'\.0+$', '', value_str)  # .0 だけの場合は削除
    return value_str
