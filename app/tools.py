
# ======================================================
# WEATHER TOOL
# ======================================================

def get_weather():
    return {
        "tool": "weather",
        "location": "Demo Location",
        "temperature": "28°C",
        "condition": "Partly cloudy",
        "source": "Demo weather tool"
    }


# ======================================================
# CALCULATOR TOOL
# ======================================================

def calculate(expression: str):
    try:
        allowed_characters = "0123456789+-*/(). "

        if not expression or not all(
            character in allowed_characters
            for character in expression
        ):
            return {
                "error": "Invalid calculator expression."
            }

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        if not isinstance(result, (int, float)):
            return {
                "error": "Invalid calculation result."
            }

        return {
            "tool": "calculator",
            "expression": expression,
            "result": result,
            "source": "Calculator tool"
        }

    except (SyntaxError, TypeError, ZeroDivisionError, ValueError):
        return {
            "error": "Could not calculate the expression."
        }


# ======================================================
# DATABASE TOOL (DEMO)
# ======================================================

EMPLOYEES = {
    "101": {
        "employee_id": "101",
        "name": "Rahul",
        "department": "Engineering",
        "designation": "Python Developer"
    },
    "102": {
        "employee_id": "102",
        "name": "Priya",
        "department": "HR",
        "designation": "HR Executive"
    }
}


def get_employee(employee_id: str):

    employee = EMPLOYEES.get(employee_id)

    if employee is None:
        return {
            "error": f"Employee {employee_id} not found."
        }

    return {
        "tool": "database",
        "data": employee,
        "source": "Demo employee database"
    }


# ======================================================
# TOOL EXECUTION
# ======================================================

def execute_tool(question: str):

    question = question.lower().strip()

    # WEATHER
    if "weather" in question:
        return get_weather()

    # CALCULATOR
    if question.startswith("calculate "):

        expression = question[len("calculate "):].strip()

        return calculate(expression)

    # DATABASE
    if "employee" in question:

        words = question.split()

        employee_id = next(
            (
                word.strip(".,?!")
                for word in words
                if word.strip(".,?!").isdigit()
            ),
            None
        )

        if employee_id is None:
            return {
                "error": "Please provide an employee ID."
            }

        return get_employee(employee_id)

    # UNKNOWN TOOL
    return {
        "error": "No matching tool found."
    }