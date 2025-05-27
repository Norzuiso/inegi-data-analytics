import re
x = "Hola"
condition = "(x == Hola) || (x != adios)"

def generate_condition_str(condition: str):
    condition = condition.replace(" ", "")
    results = []
    if not condition:
        raise ValueError("Condition cannot be empty.")
    if "&&" in condition or "||" in condition:
        split_conditions = re.split(r'\s*&&\s*|\s*\|\|\s*', condition)
        for cond in split_conditions:
            if not cond.strip():
                raise ValueError("Condition cannot contain empty segments.")
            if "(" in cond or ")" in cond:
                final_condition = cond.replace("(", "").replace(")", "")
                if "is" in final_condition:
                    final_condition = transform_condition(cond, final_condition, "==")
                    results.append(x == final_condition[1])
                elif "not" in final_condition:
                    final_condition = transform_condition(cond, final_condition, "!=")
                    results.append(x != final_condition[1])

    return True

def transform_condition(cond, final_condition, operator):
    final_condition = final_condition.replace("x", "")
    final_condition = final_condition.split(operator)
    if len(final_condition) != 1:
        raise ValueError(f"Condition segment '{cond}' is not valid. Expected format: 'x is value' or 'x is not value'.")
    return final_condition


generate_condition_str(condition)