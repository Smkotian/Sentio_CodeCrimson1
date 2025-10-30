# Install once (if needed):
# pip install pandas haversine folium ipywidgets

import pandas as pd
import ast
from haversine import haversine, Unit
import folium
import ipywidgets as widgets
from IPython.display import display, clear_output

# === Load CSV file locally ===
filename = 'blood_banks_formatted.csv'
df = pd.read_csv(filename)
# === Clean and prepare data ===
df = df.rename(columns={'O': 'O+', 'A': 'A+', 'B': 'B+', 'AB': 'AB+'})

blood_types = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
for btype in blood_types:
    if btype not in df.columns:
        df[btype] = 0
    df[btype] = pd.to_numeric(df[btype], errors='coerce').fillna(0).astype(int)

df['expiry_dates'] = df['expiry_dates'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

# === Distance computation ===
def compute_distance(loc1, loc2):
    return haversine(loc1, loc2, unit=Unit.KILOMETERS)

# === Search + map ===
def search_and_map(lat, lon, blood_type, radius):
    patient_loc = (lat, lon)
    results = []
    for _, row in df.iterrows():
        dist = compute_distance(patient_loc, (row['lat'], row['lon']))
        if dist <= radius and row[blood_type] > 0:
            results.append({
                'name': row['name'],
                'distance_km': dist,
                'units_available': row[blood_type],
                'expiry_dates': row['expiry_dates'],
                'lat': row['lat'],
                'lon': row['lon']
            })
    results = sorted(results, key=lambda x: x['distance_km'])

    fmap = folium.Map(location=patient_loc, zoom_start=9)
    folium.Marker(
        location=patient_loc,
        popup='Patient Location',
        icon=folium.Icon(color='red', icon='user')
    ).add_to(fmap)

    for res in results:
        folium.Marker(
            location=[res['lat'], res['lon']],
            popup=f"{res['name']}<br>Units: {res['units_available']}<br>Distance: {res['distance_km']:.2f} km",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(fmap)

    display(fmap)
    print(f"\nFound {len(results)} blood banks with {blood_type} within {radius} km")
    for r in results:
        print(f"{r['name']} ({r['distance_km']:.2f} km), units: {r['units_available']}, expiries: {r['expiry_dates']}")

# === Interactive widgets ===
lat_widget = widgets.FloatText(value=15.00, description='Latitude:')
lon_widget = widgets.FloatText(value=76.50, description='Longitude:')
blood_widget = widgets.Dropdown(options=blood_types, description='Blood Type:', value='A+')
radius_widget = widgets.FloatSlider(value=50, min=1, max=200, step=1, description='Radius (km):')
search_button = widgets.Button(description="Search")
output_widget = widgets.Output()

def on_search_button_clicked(b):
    with output_widget:
        clear_output()
        try:
            search_and_map(lat_widget.value, lon_widget.value, blood_widget.value, radius_widget.value)
        except Exception as e:
            print(f"An error occurred: {e}")

search_button.on_click(on_search_button_clicked)
display(lat_widget, lon_widget, blood_widget, radius_widget, search_button, output_widget)