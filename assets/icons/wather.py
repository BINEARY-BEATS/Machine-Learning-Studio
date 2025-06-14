import tkinter as tk
from tkinter import messagebox
import requests

def fetch_weather():
    city = city_entry.get()
    if not city:
        messagebox.showerror("Error", "Please enter a city name!")
        return
    
    API_KEY = "YOUR_API_KEY"  # Replace with your API key
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if data["cod"] == 200:
            weather_info = (
                f"Weather in {city}:\n"
                f"- Condition: {data['weather'][0]['description']}\n"
                f"- Temperature: {data['main']['temp']}°C\n"
                f"- Humidity: {data['main']['humidity']}%\n"
                f"- Wind Speed: {data['wind']['speed']} m/s"
            )
            result_label.config(text=weather_info)
        else:
            messagebox.showerror("Error", data["message"])
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch weather: {e}")

# GUI Setup
app = tk.Tk()
app.title("Weather App")
app.geometry("400x300")

tk.Label(app, text="Enter City:").pack(pady=5)
city_entry = tk.Entry(app, width=30)
city_entry.pack(pady=5)

fetch_button = tk.Button(app, text="Get Weather", command=fetch_weather)
fetch_button.pack(pady=10)

result_label = tk.Label(app, text="", justify="left")
result_label.pack(pady=10)

app.mainloop()