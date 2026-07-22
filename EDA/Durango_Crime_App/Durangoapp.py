import json  # ¡Importante para manejar el formato GeoJSON!
import os
import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILO
# ==========================================
st.set_page_config(
    page_title="Tablero Analítico de Incidencia Delictiva en Durango (2020)",  # 👉 Año agregado
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Estilo CSS personalizado para una interfaz limpia y moderna
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    div.block-container { padding-top: 1.5rem; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #d9534f;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. FUNCIONES DE CARGA Y LIMPIEZA DE DATOS
# ==========================================
@st.cache_data
def load_and_clean_csv(filepath: str) -> pd.DataFrame:
    """Lee ForDurangoMap.csv, corrige errores tipográficos y estandariza los IDs."""
    # Omitimos la fila 0 por las coordenadas de Excel (M, N, O...) para que la fila 1 sea el encabezado
    df = pd.read_csv(filepath, skiprows=1)

    # Limpiamos espacios en blanco alrededor de los nombres de las columnas
    df.columns = df.columns.str.strip()

    # Búsqueda automática de la columna de ID municipal (Cve#, ID, Clave, etc.)
    for col in df.columns:
        if any(
            term in str(col).lower()
            for term in ["cve", "id", "#", "clave", "code", "cod"]
        ):
            df = df.rename(columns={col: "CVEGEO"})
            break

    # Corregimos errores de dedo generados durante la exportación a Excel/CSV
    rename_map = {
        "asa de Lesiones Dolosas": "Tasa de Lesiones Dolosas",
        "asa de Robo con Violencia": "Tasa de Robo con Violencia",
    }
    df = df.rename(columns=rename_map)

    # Eliminamos filas finales vacías o las filas de 'Total General' de Excel
    df = df[df["Municipio"].notna()]
    df = df[
        ~df["Municipio"].astype(str).str.contains("Total|Labels", case=False)
    ]

    # Estandarizamos la CVEGEO a una cadena de 5 dígitos comenzando con '10' (ej. '10001')
    if "CVEGEO" in df.columns:
        df["CVEGEO"] = (
            df["CVEGEO"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.strip()
            .str.zfill(5)
        )

    # Nos aseguramos de que las columnas numéricas tengan el tipo de dato correcto (float/int)
    numeric_cols = [
        "Population",
        "Tasa Total de Delitos",
        "Tasa de Homicidio Doloso",
        "Tasa de Feminicidio",
        "Tasa de Lesiones Dolosas",
        "Tasa de Violencia Familiar",
        "Tasa de Extorsión",
        "Tasa de Robo con Violencia",
        "Tasa de Delitos con Arma de Fuego",
        "Tasa de Narcomenudeo",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


@st.cache_data
def load_spatial_data(filepath: str) -> gpd.GeoDataFrame:
    """Carga los límites geográficos y estandariza las claves municipales."""
    gdf = gpd.read_file(filepath)

    # Convertimos los nombres de las columnas a mayúsculas
    gdf.columns = [
        col.upper() if col != "geometry" else "geometry" for col in gdf.columns
    ]

    # Convertimos las coordenadas al estándar GPS (WGS84) si es necesario
    if gdf.crs is not None and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Búsqueda automática del ID en el GeoJSON y conversión a CVEGEO de 5 dígitos
    if "CVEGEO" not in gdf.columns:
        id_col = None
        # 1. Buscamos palabras clave en español e inglés (incluyendo 'codigo' / 'cod' / 'D_CODIGO')
        for col in gdf.columns:
            if col != "geometry" and any(
                term in str(col).lower()
                for term in [
                    "cve",
                    "id",
                    "code",
                    "codigo",
                    "cod",
                    "mun",
                    "clave",
                ]
            ):
                id_col = col
                break

        # 2. Si no hay coincidencia, tomamos la primera columna que no sea 'geometry'
        if not id_col:
            for col in gdf.columns:
                if col != "geometry":
                    id_col = col
                    break

        # 3. Estandarizamos los valores a una cadena de 5 dígitos (ej. '10001')
        if id_col:
            clean_vals = (
                gdf[id_col]
                .astype(str)
                .str.replace(".0", "", regex=False)
                .str.strip()
            )
            # Si el número tiene entre 1 y 3 dígitos (ej. '1' o '001'), agregamos el prefijo '10' de Durango
            gdf["CVEGEO"] = clean_vals.apply(
                lambda x: (
                    "10" + x.zfill(3) if len(x.lstrip("0")) <= 3 else x.zfill(5)
                )
            )

    return gdf


# ==========================================
# 3. BÚSQUEDA AUTOMÁTICA DE ARCHIVOS Y UNIÓN
# ==========================================
# Busca en la carpeta actual y en la subcarpeta data/ automáticamente
possible_csv_paths = ["ForDurangoMap.csv", "data/ForDurangoMap.csv"]
possible_geo_paths = ["durango_municipios.geojson", "data/durango_municipios.geojson"]

csv_path = next((p for p in possible_csv_paths if os.path.exists(p)), None)
geo_path = next((p for p in possible_geo_paths if os.path.exists(p)), None)

if not csv_path:
    st.error(
        "❌ No se encontró el archivo `ForDurangoMap.csv`. ¡Por favor colócalo dentro de tu carpeta de trabajo!"
    )
    st.stop()

if not geo_path:
    st.error("❌ No se encontró el archivo `durango_municipios.geojson`.")
    st.stop()

# Carga de datos
df_crime = load_and_clean_csv(csv_path)
gdf_boundaries = load_spatial_data(geo_path)

# Verificamos que se haya encontrado la columna CVEGEO en ambos archivos
if "CVEGEO" not in df_crime.columns:
    st.error(
        f"❌ ¡No se encontró la clave municipal en el archivo CSV! Columnas disponibles: {list(df_crime.columns)}"
    )
    st.stop()

if "CVEGEO" not in gdf_boundaries.columns:
    st.error(
        f"❌ ¡No se encontró la clave municipal en el GeoJSON! Columnas disponibles: {list(gdf_boundaries.columns)}"
    )
    st.stop()

# Unión espacial (inner join) coincidiendo las claves municipales
merged_gdf = gdf_boundaries.merge(df_crime, on="CVEGEO", how="inner")


# ==========================================
# 4. BARRA LATERAL (FILTROS Y CONTROLES)
# ==========================================
st.sidebar.title("🎛️ Controles del Tablero")
st.sidebar.markdown("Filtra y personaliza las visualizaciones espaciales.")

rate_options = [
    "Tasa Total de Delitos",
    "Tasa de Homicidio Doloso",
    "Tasa de Feminicidio",
    "Tasa de Lesiones Dolosas",
    "Tasa de Violencia Familiar",
    "Tasa de Extorsión",
    "Tasa de Robo con Violencia",
    "Tasa de Delitos con Arma de Fuego",
    "Tasa de Narcomenudeo",
]

# Selector de delito
selected_crime = st.sidebar.selectbox(
    "📊 Selecciona el Indicador Delictivo:",
    options=rate_options,
    index=0,
    help="Muestra la tasa delictiva por cada 100,000 habitantes en los municipios de Durango.",
)

# 👉 MODIFICADO: Paleta de color fijada a "Reds" permanentemente (sin menú desplegable)
color_scale = "Reds"

# Filtro por rango de población
min_pop = int(df_crime["Population"].min())
max_pop = int(df_crime["Population"].max())
pop_filter = st.sidebar.slider(
    "👥 Filtrar por Rango de Población:",
    min_value=min_pop,
    max_value=max_pop,
    value=(min_pop, max_pop),
    step=1000,
)

# Búsqueda de un municipio específico
all_municipios = ["Todos los Municipios"] + sorted(
    df_crime["Municipio"].unique().tolist()
)
selected_muni = st.sidebar.selectbox("🔍 Resaltar Municipio Específico:", options=all_municipios)

st.sidebar.markdown("---")
st.sidebar.caption("Fuentes de Datos: SESNSP e INEGI (Censo de Población y Vivienda)")

# Aplicación de los filtros a los DataFrames
filtered_df = df_crime[
    (df_crime["Population"] >= pop_filter[0])
    & (df_crime["Population"] <= pop_filter[1])
]

if selected_muni != "Todos los Municipios":
    filtered_df = filtered_df[filtered_df["Municipio"] == selected_muni]

filtered_gdf = merged_gdf[merged_gdf["CVEGEO"].isin(filtered_df["CVEGEO"])]

# ==========================================
# 5. ENCABEZADO PRINCIPAL Y MÉTRICAS (KPIs)
# ==========================================
st.title("Atlas Interactivo de Incidencia Delictiva en Durango (2020)")

st.markdown(
    """
**Plataforma de análisis espacial para la evaluación territorial de la seguridad pública.**  
Este tablero modela el comportamiento del crimen cruzando los reportes oficiales del **SESNSP** con la demografía del Censo de Población (**INEGI 2020**).

* **📊 Alcance estatal:** Cartografía y monitoreo de **8 delitos de alto impacto** a través de los **39 municipios** de Durango.
* **⚖️ Estandarización demográfica:** Las cifras se expresan en **tasas por cada 100,000 habitantes**, eliminando el sesgo poblacional para comparar equitativamente localidades rurales con zonas urbanas como Durango o Gómez Palacio.
* **🎛️ Navegación:** Utiliza el menú lateral para seleccionar un indicador delictivo, filtrar por rango de habitantes o aislar un municipio específico.
"""
)

# Aquí continúan tus columnas (col1, col2, col3, col4)...

# Aquí continúan tus columnas (col1, col2, col3, col4)...

col1, col2, col3, col4 = st.columns(4)

avg_rate = df_crime[selected_crime].mean()
max_rate_row = df_crime.loc[df_crime[selected_crime].idxmax()]
total_pop = df_crime["Population"].sum()

with col1:
    st.metric(
        label=f"Promedio Estatal ({selected_crime.replace('Tasa de ', '').replace('Tasa ', '')})",
        value=f"{avg_rate:.1f}",
    )

with col2:
    st.metric(
        label="Municipio con Tasa Más Alta",
        value=f"{max_rate_row['Municipio']}",
        delta=f"{max_rate_row[selected_crime]:.1f} / 100k",
        delta_color="inverse",
    )

with col3:
    st.metric(label="Población Total del Estado", value=f"{int(total_pop):,}")

with col4:
    st.metric(
        label="Municipios Mostrados",
        value=f"{len(filtered_df)} / {len(df_crime)}",
    )

st.markdown("---")

# ==========================================
# 6. MAPA INTERACTIVO Y GRÁFICAS DEL TOP 10
# ==========================================
map_col, chart_col = st.columns([3, 2])

with map_col:
    st.subheader(f"🗺️ Distribución Espacial: {selected_crime}")

    # Diagnóstico: muestra cuántos municipios se enlazaron correctamente
    st.caption(
        f"Se vincularon con éxito **{len(filtered_gdf)}** de **{len(df_crime)}** municipios para su cartografía."
    )

    if not filtered_gdf.empty:
        # 1. Aseguramos que las coordenadas estén en formato estándar GPS Lat/Lon (EPSG:4326)
        if filtered_gdf.crs is not None and filtered_gdf.crs != "EPSG:4326":
            filtered_gdf = filtered_gdf.to_crs("EPSG:4326")

        # 2. Asignamos CVEGEO como el índice para que Plotly grafique directamente sin usar JSON
        map_df = filtered_gdf.set_index("CVEGEO")

        # Construcción del mapa coroplético de Mapbox con Plotly (Fijado a OpenStreetMap)
        fig_map = px.choropleth_mapbox(
            map_df,
            geojson=map_df.geometry,
            locations=map_df.index,  # Conecta nativamente con el índice de map_df
            color=selected_crime,
            hover_name="Municipio",
            hover_data={
                "Population": ":,",
                "Tasa Total de Delitos": ":.1f",
                "Tasa de Homicidio Doloso": ":.1f",
                "Tasa de Violencia Familiar": ":.1f",
                selected_crime: ":.2f",
            },
            color_continuous_scale=color_scale,
            mapbox_style="open-street-map",  # 👉 MODIFICADO: Estilo fijo sin opciones
            zoom=5.5,
            center={"lat": 24.8, "lon": -104.8},
            opacity=0.75,
            labels={selected_crime: "Tasa / 100k hab.", "Population": "Población"},
        )

        fig_map.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=500,
            coloraxis_colorbar=dict(
                title="Tasa / 100k",
                thicknessmode="pixels",
                thickness=15,
                lenmode="pixels",
                len=300,
            ),
        )

        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning(
            "⚠️ Ningún municipio coincide con los filtros aplicados en la barra lateral."
        )

with chart_col:
    st.subheader("📈 Municipios con Mayor Incidencia")

    max_muni = len(df_crime) if len(df_crime) > 0 else 39
    top_n = st.slider(
        "Mostrar Top N Municipios:", min_value=1, max_value=max_muni, value=10
    )
    
    top_df = (
        filtered_df.sort_values(by=selected_crime, ascending=False)
        .head(top_n)
        .sort_values(by=selected_crime, ascending=True)
    )

    fig_bar = px.bar(
        top_df,
        x=selected_crime,
        y="Municipio",
        orientation="h",
        text=selected_crime,
        color=selected_crime,
        color_continuous_scale=color_scale,
        labels={selected_crime: "Tasa por cada 100k hab.", "Municipio": ""},
    )

    fig_bar.update_traces(
        texttemplate="%{text:.1f}", textposition="outside", showlegend=False
    )
    
    chart_height = max(500, top_n * 22)
    fig_bar.update_layout(
        height=chart_height,
        margin={"r": 20, "t": 10, "l": 0, "b": 0},
        coloraxis_showscale=False,
    )

    st.plotly_chart(fig_bar, use_container_width=True)
