def classify_severity(value):
    if value < 1:
        return "Low"
    elif value < 3:
        return "Moderate"
    elif value < 6:
        return "High"
    else:
        return "Very High"

def detect_trend(values):
    if values[-1] > values[0]:
        return "Increasing"
    elif values[-1] < values[0]:
        return "Decreasing"
    return "Stable"
