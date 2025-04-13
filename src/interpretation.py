import numpy as np
import math as math
import requests

openWeatherToken = "325b7de412ffd1e7d1866995d1779a93"
base_url = "http://api.openweathermap.org/data/2.5/weather?"

def generateRisk(density, location, eventType, percentExitsBlocked):
    weights = {
        "concert":    {"density": 0.30, "heat": 0.20, "exits": 0.20},
        "protest":    {"density": 0.25, "heat": 0.15, "exits": 0.25},
        "pilgrimage": {"density": 0.40, "heat": 0.20, "exits": 0.2},
    }

    weights = weights.get(eventType, None)
    if weights is None:
        raise ValueError(f"Unknown event type: {eventType}")
        
    def getTempHumidity(location):
        complete_url = base_url + "q=" + location + "&appid=" + openWeatherToken
        response = requests.get(complete_url)
        data = response.json()
        if data["cod"] != "404":
            main = data["main"]
            temperature = main["temp"] - 273.15
            humidity = main["humidity"]
        return temperature + humidity

    def normalize(value, minValue, maxValue):
        return max(0, min(100, (value - minValue) / (maxValue - minValue) * 100))
    
    densityScore = 0
    if density >= 5:
        densityScore = 30
    elif density >= 4:
        densityScore = 20
    elif density >= 3:
        densityScore = 10

    heat_score = getTempHumidity(location) 

    exit_score = percentExitsBlocked

    total = (
        densityScore * weights["density"] +
        normalize(heat_score, 20, 30) * weights["heat"] +
        normalize(exit_score, 10, 50) * weights["exits"]
    )

    def identifyRisk(score):
        if score >= 85:
            return "EXTREME RISK (Evacuate now)"
        elif score >= 60:
            return "HIGH RISK (Immediate action)"
        elif score >= 30:
            return "MODERATE RISK (Monitor closely)"
        else:
            return "LOW RISK (Normal operations)"

    def suggest_mitigation(score, density, heat_index):
        suggestions = []
        if score >= 60:
            suggestions.append("HALT EVENT: Stop inflow of people.")
        if density >= 4:
            suggestions.append("OPEN EXITS: Redirect crowd to unused exits.")
        if heat_index >= 32:
            suggestions.append("COOLING STATIONS: Deploy water/misting fans.")
        if score >= 85:
            suggestions.append("EVACUATE: Use loudspeakers for orderly exit.")
        if len(suggestions) != 0:
            return suggestions 
        else: return ["No critical actions needed."]
    # Returns in the format of -
    """
    Total Score: {total}
    Risk Level: {identifyRisk(total)}
    Suggestions: {suggest_mitigation(total, density, heat_score)}
    """
    return {
        "total_score": total,
        "risk_level": identifyRisk(total),
        "suggestions": suggest_mitigation(total, density, heat_score),
        "heat_score": heat_score,
        "density_score": densityScore,
        "exit_score": exit_score
    }

