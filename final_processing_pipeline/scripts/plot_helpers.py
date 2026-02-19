import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display
from typing import Tuple, Optional, Union, Dict, Any
import pandas as pd
import folium
from pyproj import CRS, Transformer


## ----------------------------------------------------------------------
## II. Statische RGB-Funktion (kann einzeln aufgerufen werden)
## ----------------------------------------------------------------------


def plot_rgb(
    ds: xr.Dataset,
    timestep: int,
    time_dim: str = "time_sentinel_2_l2a",
    bands: Tuple[str, str, str] = ("B04", "B03", "B02"),
    stretch_pct: Tuple[int, int] = (2, 98),
    figsize: Tuple[int, int] = (10, 6),
    cloud_comp: bool = False,
    title_prefix: Optional[str] = "Timestep",
) -> Union[None, str]:
    """
    Plots a static true-color (RGB) composite for a specific timestep
    from a Sentinel-2 xarray.Dataset. Optionally plots a cloud mask comparison.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have a time coordinate `time_dim` and DataArrays for the three bands.
    timestep : int
        The time index to plot (must be valid for ds[time_dim]).
    time_dim : str
        Name of the time dimension (default 'time_sentinel_2_l2a').
    bands : tuple of str
        The (red, green, blue) band names in ds (default ('B04','B03','B02')).
    stretch_pct : tuple of int
        Percentiles for contrast stretch (default 2nd to 98th).
    figsize : tuple
        Matplotlib figure size.
    cloud_comp : bool
        If True, plots the RGB image and the cloud mask comparison side-by-side.
    title_prefix : str or None
        Prefix for the figure title (used by interactive function).

    Returns
    -------
    None or str
        None on success, or an error message string if the timestep is invalid.
    """

    # Helper function to percentile‐stretch one (y,x) slice
    def stretch_arr(band):
        arr = band.compute().astype("float32")
        p2, p98 = np.percentile(arr, stretch_pct)
        return np.clip((arr - p2) / (p98 - p2), 0, 1)

    # 1. Input Validation
    if not 0 <= timestep < len(ds.coords[time_dim]):
        return f"Timeindex {timestep} out of range for dimension '{time_dim}' (size {len(ds.coords[time_dim])})."

    # 2. Subset data for the given timestep
    subset = ds.isel({time_dim: timestep})
    time_value_raw = subset[time_dim].item()

    time_formatted = pd.to_datetime(time_value_raw).strftime("%Y-%m-%d")

    try:
        r, g, b = subset[bands[0]], subset[bands[1]], subset[bands[2]]
    except KeyError as e:
        return f"One or more bands not found in dataset: {e}"

    # 3. Create RGB array (stretched)
    rgb = np.dstack(
        [stretch_arr(r).values, stretch_arr(g).values, stretch_arr(b).values]
    )

    # 4. Plotting Logic

    if cloud_comp:
        # --- Cloud Comparison Logic ---
        try:
            scl = subset["SCL"]
            cloud_mask = subset["cloud_mask"]
        except KeyError as e:
            return f"Required variable for cloud comparison missing: {e}. Set cloud_comp=False or check dataset."

        # Binary masks
        is_cloud_scl = scl.isin([3, 8, 9, 10])  # cloud classes SCL
        is_external_cloud = cloud_mask.isin([1, 2, 3])  # cloud classes cloud_mask

        # Calculate difference and agreement masks for the overlay
        scl_only_mask = (
            (is_cloud_scl & ~is_external_cloud).astype(float).values.squeeze()
        )
        external_only_mask = (
            (is_external_cloud & ~is_cloud_scl).astype(float).values.squeeze()
        )
        agreement_mask = (
            (is_external_cloud & is_cloud_scl).astype(float).values.squeeze()
        )

        # Create the RGB overlay (Red=SCL Only, Green=External Only, Blue=Agreement)
        overlay = np.dstack(
            [
                scl_only_mask,
                external_only_mask,
                agreement_mask,
            ]
        )

        # Plotting side by side (2 plots)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        # Left Plot: RGB ONLY
        ax1.imshow(rgb)
        ax1.set_title("True Color (RGB) Image", fontsize=12)
        ax1.axis("off")

        # Right Plot: RGB + Mask Comparison Overlay
        ax2.imshow(rgb)
        ax2.imshow(overlay, alpha=0.5)

        ax2.set_title(
            "Mask Comparison\n" "Red: SCL Only | Green: External Only | Blue: Both",
            fontsize=12,
        )
        ax2.axis("off")

        fig.suptitle(f"{title_prefix}: {time_formatted}", fontsize=14, y=1.02)

    else:
        # Plotting RGB ONLY (1 plot)
        fig, ax1 = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

        ax1.imshow(rgb)
        ax1.set_title(
            f"True Color (RGB) Image\n{title_prefix}: {time_formatted}", fontsize=14
        )
        ax1.axis("off")

    plt.show()
    return None


## ----------------------------------------------------------------------
## III. Interaktive RGB-Funktion (nutzt plot_rgb intern)
## ----------------------------------------------------------------------


def interactive_s2_rgb(
    ds: xr.Dataset,
    time_dim: str = "time_sentinel_2_l2a",
    bands: Tuple[str, str, str] = ("B04", "B03", "B02"),
    stretch_pct: Tuple[int, int] = (2, 98),
    figsize: Tuple[int, int] = (14, 6),
    slider_width: str = "900px",
    cloud_comp: bool = False,
):
    """
    Builds an interactive slider to browse true-color (RGB) composites
    from a Sentinel-2 xarray.Dataset, with an optional cloud mask comparison.

    This function calls plot_rgb for the actual plotting.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have a time coordinate `time_dim` and DataArrays for the three bands.
    time_dim : str
        Name of the time dimension (default 'time_sentinel_2_l2a').
    bands : tuple of str
        The (red, green, blue) band names in ds.
    stretch_pct : tuple of int
        Percentiles for contrast stretch (default 2nd to 98th).
    figsize : tuple
        Matplotlib figure size.
    slider_width : str
        CSS width for the slider widget.
    cloud_comp : bool
        If True, plots the RGB image and the cloud mask comparison side-by-side.
        If False (default), plots only the RGB image.
    """
    # Grab times and build slider labels
    times = list(ds[time_dim].values)
    # The slider needs the time value, but the index (i) for the subsetting logic
    labels = [
        (f"{i}: {np.datetime_as_string(t, unit='D')}", i) for i, t in enumerate(times)
    ]

    slider = widgets.SelectionSlider(
        options=labels, description="Timestep Index", layout={"width": slider_width}
    )

    @widgets.interact(i=slider)
    def _plot_rgb_interactive(i: int):
        # The interactive function now just passes the index (i) to the static function
        # The slider label (text) is now generated dynamically by the static function
        result = plot_rgb(
            ds=ds,
            timestep=i,
            time_dim=time_dim,
            bands=bands,
            stretch_pct=stretch_pct,
            figsize=figsize,
            cloud_comp=cloud_comp,
            # Pass the slider label to be used as a title prefix
            title_prefix=f"Index {i}: {np.datetime_as_string(times[i], unit='D')}",
        )
        if isinstance(result, str):
            print(f"Plotting Error: {result}")

    # The function returns the display object (the interactive widget)
    return display(_plot_rgb_interactive)


## ----------------------------------------------------------------------
## IV. Landcover Plot Funktion
## ----------------------------------------------------------------------


# --- ESA WorldCover 10m 2020 Klassifizierung ---
# Die Codes und Farben (in hex) sind Standard für diese Klassifizierung.
LC_MAPPING: Dict[int, str] = {
    10: "Tree cover",  # Dunkelgrün
    20: "Shrubland",  # Helleres Grün
    30: "Grassland",  # Helles leuchtendes Grün
    40: "Cropland",  # Bräunlich/Beige
    50: "Built-up",  # Dunkelgrau
    60: "Bare/Sparse vegetation",  # Mittelgrau
    70: "Snow and ice",  # Sehr helles Weißblau
    80: "Permanent water bodies",  # Kräftiges Blau
    90: "Herbaceous wetland",  # Mittelgrün
    100: "Moss and lichen",  # Silber/Grau
    1: "Unknown/No Data",  # Weiß
}

# Farbwerte (HEX) für die neue Darstellung
LC_COLOR_HEX: Dict[int, str] = {
    10: "#008000",  # Tree cover
    20: "#80C064",  # Shrubland
    30: "#B4FF96",  # Grassland
    40: "#C8AA5A",  # Cropland
    50: "#555555",  # Built-up
    60: "#A9A9A9",  # Bare/Sparse vegetation
    70: "#F0F8FF",  # Snow and ice
    80: "#0000FF",  # Permanent water bodies
    90: "#009664",  # Herbaceous wetland
    100: "#C0C0C0",  # Moss and lichen
    1: "#FFFFFF",  # Unknown/No Data
}
LC_CLASSES = sorted(LC_MAPPING.keys())
LC_COLORS = [LC_COLOR_HEX[code] for code in LC_CLASSES]
LC_LABELS = [LC_MAPPING[code] for code in LC_CLASSES]


def plot_landcover(
    ds: xr.Dataset,
    lc_var: str = "ESA_LC",
    time_dim_lc: str = "time_esa_worldcover",
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plots the Land Cover (LC) map for the first timestep using ESA WorldCover
    classification and creates a custom legend.

    Parameters
    ----------
    ds : xarray.Dataset
        Das Dataset, das die LC-Variable enthält.
    lc_var : str
        Name der Landcover-Variablen (default 'ESA_LC').
    time_dim_lc : str
        Name der Zeitdimension für die LC-Variable (default 'time_esa_worldcover').
    figsize : tuple
        Matplotlib figure size.
    """

    # 1. Datenvalidierung und Subset
    if lc_var not in ds:
        print(f"Error: Landcover variable '{lc_var}' not found in the dataset.")
        return

    # Da LC statisch ist, wähle den ersten Zeitschritt
    try:
        lc_snapshot = ds[lc_var].isel({time_dim_lc: 0}).compute()
    except KeyError:
        # Fallback, falls die Dimension fehlt (z.B. wenn LC bereits 2D ist)
        lc_snapshot = ds[lc_var].compute()

    # 2. Plotting Setup
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 3. Erstellung von Colormap und Normalisierung
    cmap = mcolors.ListedColormap(LC_COLORS)

    # Erstelle Grenzen für die diskreten LC Codes
    # (z.B. für Code 10: Grenzen 5 und 15; für Code 20: Grenzen 15 und 25)
    bounds = np.array(LC_CLASSES)
    bounds = np.insert(bounds, 0, bounds[0] - 5)  # Füge den ersten Rand hinzu
    bounds = bounds[:-1] + np.diff(bounds) / 2

    # Füge den letzten Rand hinzu
    bounds = np.append(bounds, bounds[-1] + 10)

    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 4. Plotten der LC-Karte
    ax.imshow(lc_snapshot.values, cmap=cmap, norm=norm, interpolation="nearest")

    # 5. Legende erstellen (Patch-Objekte verwenden, um die Farben zuzuordnen)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=cmap(norm(code))) for code in LC_CLASSES]

    ax.legend(
        handles,
        LC_LABELS,
        title="ESA WorldCover Classes",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,
    )

    # 6. Formatierung
    ax.set_title(f"Landcover Map (LC Code: {lc_var})", fontsize=14)
    ax.set_xlabel("X-Coordinate Index")
    ax.set_ylabel("Y-Coordinate Index")
    ax.axis("off")  # Deaktiviere Achsen-Ticks für bessere Visualisierung

    plt.tight_layout()
    plt.show()


## ----------------------------------------------------------------------
## V. Open Street Maps Plot Funktion
## ----------------------------------------------------------------------


def show_data_on_interactive_map(ds, edge_size_m=10000.0):
    """
    Transforms UTM coordinates to WGS84 and displays the location on an interactive map.
    The function provides both a detailed street map (with borders/cities) and a
    satellite view, which the user can switch between using the Layer Control.

    Args:
        ds (xarray): Data set
        edge_size_m (float): The side length of the bounding box in meters.

    Returns:
        folium.Map: A Folium map object.
    """

    # 1. Get information from ds
    epsg_code = ds.epsg
    central_x = ds.central_x
    central_y = ds.central_y

    # 2. Transform UTM (e.g., EPSG:32634) to WGS84 (EPSG:4326)
    # WGS84 (4326) is the standard Latitude/Longitude system used by web maps.
    transformer = Transformer.from_crs(CRS(epsg_code), CRS(4326))
    latitude, longitude = transformer.transform(central_x, central_y)

    # 2. Create the Folium Map
    # Start with 'OpenStreetMap', which provides rich context (borders, cities, roads).
    m = folium.Map(location=[latitude, longitude], zoom_start=13, tiles="OpenStreetMap")

    # Add a Satellite tile layer as an alternative option.
    # The user can click the Layer Control icon (top right of the map) to switch views.
    folium.TileLayer("Esri.WorldImagery", name="Satellite View").add_to(m)

    # 3. Add the central marker
    folium.Marker(
        [latitude, longitude],
        tooltip="Central Point",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    # 4. Calculate and add the Bounding Box
    half_size = edge_size_m / 2

    # Calculate corners in UTM
    utm_corners = [
        (central_x - half_size, central_y - half_size),
        (central_x + half_size, central_y + half_size),
    ]

    # Transform UTM corners for the bounding box (bottom-left and top-right)
    lat_min, lon_min = transformer.transform(utm_corners[0][0], utm_corners[0][1])
    lat_max, lon_max = transformer.transform(utm_corners[1][0], utm_corners[1][1])

    # Add the bounding box as a Rectangle
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color="yellow",
        weight=3,
        fill=False,
        tooltip=f"Data Coverage Area ({edge_size_m/1000}km x {edge_size_m/1000}km)",
    ).add_to(m)

    # 5. Add a Layer Control to enable switching between base map layers
    folium.LayerControl().add_to(m)

    return m


## ----------------------------------------------------------------------
## V. Find Cloud free indices for S2
## ----------------------------------------------------------------------


def find_cloud_free_indices(ds, threshold=0.99):
    """
    Findet die Indizes der Sentinel-2 Zeitschritte, die nahezu wolkenfrei sind.
    Basiert auf der 'mask_complete_quality' Variable.
    """
    # 1. Räumliches Mittel berechnen (Qualitätsrate pro Zeitschritt)
    quality_means = ds.mask_complete_quality.mean(dim=("x", "y"))

    # 2. Indizes finden, die die Bedingung erfüllen
    indices = np.where(quality_means > threshold)[0]

    # 3. Anzahl berechnen
    count = len(indices)

    print(f"Gefundene wolkenfreie Zeitschritte (> {threshold*100}%): {count}")

    return indices


##
## VI. Plot Mean variable over temporal dimension
##


def plot_spatial_analysis_map(
    ds: Any,
    variable_name: str,
    time_dim: str,
    title: str = "Spatial Analysis Map",
    cmap_name: str = "viridis",
    smart_scaling: bool = False,
) -> None:
    """
    Plots a 2D map showing the mean of a variable over the entire time series.
    Applies smart scaling based on IQR to ensure the plot is not dominated by outliers.

    Args:
        ds (Any): The xarray Dataset.
        variable_name (str): Name of the variable to analyze (e.g., 'missing_overall', 'vv_db').
        time_dim (str): Name of the time dimension (e.g., 'time_sentinel_2_l2a').
        title (str): Title of the plot.
        cmap_name (str): Colormap name. Uses 'RdYlGn_r' for fractions (0-1).
        smart_scaling (bool): If True, uses the IQR method (Q5 and Q95) for robust color limits.
    """

    # 1. Berechne die räumliche Karte (Mittelwert über die Zeitdimension)
    # Dies ist die 2D-Karte, die geplottet wird.
    spatial_map = ds[variable_name].mean(dim=time_dim)

    # 2. Intelligente Skalierung (IQR-Methode)
    # Wenn die Werte bereits eine Fraktion sind (0-1), behalten wir die Skala bei.
    if (
        spatial_map.max().values <= 1.0
        and spatial_map.min().values >= 0.0
        and not smart_scaling
    ):
        vmin = 0.0
        vmax = 1.0
        cmap_name = "RdYlGn_r"
    else:
        # Führe eine robuste Skalierung durch, um Ausreißer zu ignorieren
        # Dies ist das "smarte" Skalieren: Schließe die extremsten 5% der Daten aus.
        if smart_scaling or not (
            spatial_map.max().values <= 1.0 and spatial_map.min().values >= 0.0
        ):
            # Entferne NaNs für die Quantilberechnung
            data_valid = spatial_map.values[~np.isnan(spatial_map.values)]

            # Verwende das 5. und 95. Perzentil als Farb-Grenzen
            vmin = np.percentile(data_valid, 5)
            vmax = np.percentile(data_valid, 95)
        else:
            # Fallback auf die tatsächlichen Min/Max-Werte
            vmin = spatial_map.min().values
            vmax = spatial_map.max().values

    # 3. Plotting der 2D-Karte
    fig, ax = plt.subplots(figsize=(9, 8))

    plot_handle = spatial_map.plot.imshow(
        ax=ax,
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
        cbar_kwargs={
            "label": f"Temporal Mean of {variable_name} ({vmin:.2f} to {vmax:.2f})",
            "extend": "both" if smart_scaling else "neither",
        },
    )

    plot_handle.colorbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])

    # Titel und Labels
    ax.set_title(f"{title}: Mean of {variable_name} over Time", fontsize=10)
    ax.set_xlabel("X-Coordinate", fontsize=10)
    ax.set_ylabel("Y-Coordinate", fontsize=10)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()
