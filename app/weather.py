import requests
import json

api_key = "b1f9041f1812f89b3ac75f2a9ac9d0a8"
city = "Adilabad"
url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}"

response = requests.get(url)
data = response.json()


# print(data)


forecast_entries = data.get('list', [])

# Get the current date
current_date = None
if forecast_entries:
    current_date = forecast_entries[0]['dt_txt'].split()[0]  # Extract date from timestamp


next_4_days_forecast = []
for entry in forecast_entries:
    date, time = entry['dt_txt'].split()
    if date != current_date:
        current_date = date
        next_4_days_forecast.append(entry)

    if len(next_4_days_forecast) >= 4:
        break


for entry in next_4_days_forecast:
    date, time = entry['dt_txt'].split()
    temperature = entry['main']['temp']
    weather_description = entry['weather'][0]['description']
    print(f"Date: {date}, Time: {time}, Temperature: {temperature} K, Weather: {weather_description}")