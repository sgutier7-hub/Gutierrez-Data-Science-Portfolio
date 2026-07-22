import os
import sys
import geopandas as gpd

print("⏳ Hunting for the true 39 municipality boundaries of Durango...")

# Reliable GitHub repositories hosting official Mexican Municipal boundaries
urls = [
    "https://raw.githubusercontent.com/lapanquecita/remesas/main/assets/municipios.json",
    "https://raw.githubusercontent.com/angelozamora/mexico-geojson/master/municipios.json",
    "https://raw.githubusercontent.com/diegovalle/mx-geojson/master/municipios.json",
    "https://raw.githubusercontent.com/carmonaluis/mexico-geojson/master/municipios.json",
    "https://raw.githubusercontent.com/bto/mexico-geojson/master/municipios.json",
]

durango_gdf = None

for url in urls:
    try:
        repo_name = url.split("/")[3]
        print(f"👉 Checking '{repo_name}' repository...")
        gdf = gpd.read_file(url)

        # Standardize column names to uppercase
        gdf.columns = [
            col.upper() if col != "geometry" else col for col in gdf.columns
        ]

        # Filter specifically for Durango (State Code '10' or by State Name)
        if "ESTADO" in gdf.columns:
            temp_gdf = gdf[gdf["ESTADO"].astype(str).str.zfill(2) == "10"].copy()
        elif "CVE_ENT" in gdf.columns:
            temp_gdf = gdf[
                gdf["CVE_ENT"].astype(str).str.zfill(2) == "10"
            ].copy()
        elif "STATE_CODE" in gdf.columns:
            temp_gdf = gdf[
                gdf["STATE_CODE"].astype(str).str.zfill(2) == "10"
            ].copy()
        elif "NOM_ENT" in gdf.columns:
            temp_gdf = gdf[
                gdf["NOM_ENT"].str.contains("Durango", case=False, na=False)
            ].copy()
        else:
            temp_gdf = gdf.copy()

        # THE CRITICAL VALIDATION: Does it have exactly 39 municipalities?
        if len(temp_gdf) == 39:
            print(f"✅ Verified! Found exactly 39 municipalities in {repo_name}!")
            durango_gdf = temp_gdf
            break
            
    except Exception as e:
        continue

if durango_gdf is None:
    print(
        "\n❌ Could not download the 39 municipalities. Please check your internet connection."
    )
    sys.exit(1)

# Ensure the municipality ID (CVEGEO) is cleanly formatted as '10001', '10002'...
if "CVEGEO" not in durango_gdf.columns:
    if "CVE_MUN" in durango_gdf.columns:
        durango_gdf["CVEGEO"] = "10" + durango_gdf["CVE_MUN"].astype(
            str
        ).str.zfill(3)
    elif "MUN_CODE" in durango_gdf.columns:
        durango_gdf["CVEGEO"] = "10" + durango_gdf["MUN_CODE"].astype(
            str
        ).str.zfill(3)
    elif "ID" in durango_gdf.columns:
        durango_gdf["CVEGEO"] = durango_gdf["ID"].astype(str).str.zfill(5)

# Target location for file save
output_filename = "data/durango_municipios.geojson"

# Build the directory path if it's missing before trying to use to_file
output_dir = os.path.dirname(output_filename)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

try:
    durango_gdf.to_file(output_filename, driver="GeoJSON")
except Exception as e:
    # Fallback to current directory if folder writing fails
    output_filename = "durango_municipios.geojson"
    durango_gdf.to_file(output_filename, driver="GeoJSON")

print(
    f"\n🎉 SUCCESS! Saved the true {len(durango_gdf)} municipalities to `{output_filename}`."
)
