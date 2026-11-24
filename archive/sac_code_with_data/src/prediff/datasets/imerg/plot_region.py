import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_region_on_map(data, lat_grid, lon_grid, center_lat, center_lon, size_deg=2.0):
    # Define extent
    half_size = size_deg / 2
    lat_min, lat_max = center_lat - half_size, center_lat + half_size
    lon_min, lon_max = center_lon - half_size, center_lon + half_size

    # Create a mask for the selected region
    region_mask = (
        (lat_grid >= lat_min) & (lat_grid <= lat_max) &
        (lon_grid >= lon_min) & (lon_grid <= lon_max)
    )

    # Masked data and coordinates
    region_data = np.where(region_mask, data, np.nan)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Plot data
    im = ax.pcolormesh(lon_grid, lat_grid, region_data, cmap='Blues', shading='auto')

    # Add map features
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor='none')
    ax.add_feature(cfeature.RIVERS, edgecolor='blue')

    # Optional: Add city names
    ax.set_title(f"Region around ({center_lat}, {center_lon})")

    plt.colorbar(im, ax=ax, orientation='vertical', label='Precipitation (mm/hr)')
    plt.tight_layout()
    plt.show()

