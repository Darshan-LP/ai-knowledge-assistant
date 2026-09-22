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

        if not all(
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

        return {
            "tool": "calculator",
            "expression": expression,
            "result": result,
            "source": "Calculator tool"
        }

    except Exception:

        return {
            "error": "Could not calculate the expression."
        }


# ======================================================
# TOOL EXECUTION
# ======================================================

def execute_tool(question: str):

    question = question.lower().strip()

    # --------------------------------------------------
    # WEATHER
    # --------------------------------------------------

    if "weather" in question:

        return get_weather()

    # --------------------------------------------------
    # CALCULATOR
    # --------------------------------------------------

    if "calculate" in question:

        expression = (
            question
            .replace("calculate", "")
            .strip()
        )

        if expression:

            return calculate(
                expression
            )

    # --------------------------------------------------
    # NO TOOL
    # --------------------------------------------------

    return {
        "error": "No matching tool found."
    }