from datetime import datetime


# ======================================================
# WEATHER TOOL
# ======================================================

def get_weather():

    return {
        "location": "Demo Location",
        "temperature": "28°C",
        "condition": "Partly cloudy",
        "source": "Demo weather tool"
    }


# ======================================================
# TOOL EXECUTION
# ======================================================

def execute_tool(question: str):

    question = question.lower().strip()

    if "weather" in question:

        return get_weather()

    return {
        "error": "No matching tool found."
    }