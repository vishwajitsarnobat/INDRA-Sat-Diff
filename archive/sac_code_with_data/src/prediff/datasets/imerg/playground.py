import h5py as hf
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# from mpl_toolkits.basemap import Basemap
# import matplotlib.colors as mcolors
# import os
# import multiprocessing
# from datetime import datetime, timedelta

# # plotting and saving all available files
# def plot_func(fpath, shape_file, save_plot_path, correct_values=None):
#     try:
#         with hf.File(fpath, 'r') as f:
#             date_time = convert_minutes_to_datetime(f["time"][0])
#             filename = os.path.basename(fpath).split('.')[0]

#             date = date_time.date()
#             time = date_time.time()
#             title_date_time = f"Precipitation on {date} at {time}"
            
#             year = date_time.year
#             month = date_time.month
            
#             dir_name = f"{save_plot_path}/{year}/{month}"
#             os.makedirs(dir_name, exist_ok=True)
#             save_file = f"{dir_name}/{filename}.png"
#             if not os.path.exists(save_file):
#                 # Create a colormap with white for zero values
#                 cmap = plt.cm.rainbow  # Start with a continuous colormap
#                 cmap_list = cmap(np.linspace(0, 1, 256))  # Convert it to an array of RGBA values
#                 cmap_list[0] = [1.0, 1.0, 1.0, 1.0]  # Set the first color (for zero values) to white
#                 custom_cmap = mcolors.ListedColormap(cmap_list)
    
#                 # Normalize between 0 and 100, explicitly mapping 0 to white
#                 norm = mcolors.Normalize(vmin=0, vmax=100)
                
#                 hem = f['HEM'][:]
#                 if correct_values is not None:
#                     lat, lon = correct_values["Latitude"], correct_values["Longitude"]
#                 else:
#                     lat, lon = f['Latitude'][:], f['Longitude'][:]
#                 lat, lon, hem = lat.flatten(), lon.flatten(), hem.flatten()
#                 # print(len(lat), len(lon))
                
#                 scale_factor = 0.009999999776482582
#                 invalid_value = 32767 * scale_factor
#                 lat_corrected = lat * scale_factor
#                 lon_corrected = lon * scale_factor
#                 valid_mask = (lat_corrected != invalid_value) & (lon_corrected != invalid_value) & (hem >= 0)
    
#                 lat_valid = lat_corrected[valid_mask]
#                 lon_valid = lon_corrected[valid_mask]
#                 hem_valid = hem[valid_mask]
                
#                 hem_valid = np.clip(hem_valid, 0, 100)
    
#                 if len(lat_valid) == 0 or len(lon_valid) == 0 or len(hem_valid) == 0:
#                     print(f"Warning: No valid data to plot for {fpath}.")
#                     return

#                 # code to find the same is below
#                 min_lat, max_lat = -81.03999818861485, 81.03999818861485
#                 min_lon, max_lon = -7.149999840185046, 155.14999653212726

#                 plt.figure(figsize=(14, 9))
#                 m = Basemap(projection='cyl',
#                             llcrnrlat=min_lat, urcrnrlat=max_lat,
#                             llcrnrlon=min_lon, urcrnrlon=max_lon,
#                             resolution='i')
                
#                 # Add coastlines, countries, and state boundaries
#                 m.drawcoastlines()
#                 m.drawcountries()
#                 m.drawstates()

#                 # Draw parallels and meridians for the complete image
#                 m.drawparallels(np.arange(-90, 91, 10), labels=[1, 0, 0, 0], linewidth=0.5, color='gray')
#                 m.drawmeridians(np.arange(-180, 181, 10), labels=[0, 0, 0, 1], linewidth=0.5, color='gray')
    
#                 # Read and plot shapefile for additional features (e.g., India boundaries)
#                 m.readshapefile(shape_file.split('.')[0], 'india', linewidth=1.0)
    
#                 x, y = m(lon_valid, lat_valid)
#                 sc = m.scatter(x, y, c=hem_valid, cmap=custom_cmap, norm=norm, s=10, edgecolor='none')
    
#                 # Create a continuous colorbar
#                 cbar = plt.colorbar(sc)
#                 cbar.set_label('Precipitation Intensity (mm/h)')
                
#                 plt.title(title_date_time)
#                 plt.savefig(save_file, dpi=300, bbox_inches='tight')
#                 print("Plot saved successfully at: ", save_file)
#                 plt.close()
#             else:
#                 print("File already exists! No new plot created!")
                
#     except Exception as e:
#         print(f"Error processing file {fpath}: {e}")

# def plot_map(shapefile_path_base, lon, lat):

#     lats = [34.05, 36.17, 39.74, 40.71, 38.90]
#     lons = [-118.24, -115.14, -104.99, -74.01, -77.04]

#     min_lon = lon[0]
#     max_lon = lon[1]
#     min_lat = lat[0]
#     max_lat = lat[1]

#     print("Setting up map...")
#     plt.figure(figsize=(10, 8))

#     m = Basemap(
#         projection='cyl', # Cylindrical Equidistant
#         llcrnrlat=min_lat,
#         urcrnrlat=max_lat,
#         llcrnrlon=min_lon,
#         urcrnrlon=max_lon,
#         resolution='l' # Low resolution ('c', 'l', 'i', 'h', 'f')
#     )

#     print("Drawing map boundary...")
#     m.drawmapboundary(fill_color='aqua')
#     m.drawparallels(np.arange(min_lat, max_lat + 1, 10.), labels=[1,0,0,0], fontsize=10) # Labels on left
#     m.drawmeridians(np.arange(min_lon, max_lon + 1, 10.), labels=[0,0,0,1], fontsize=10) # Labels on bottom
#     # Optionally, fill continents if shapefile doesn't cover everything
#     # m.fillcontinents(color='coral', lake_color='aqua')

#     print(f"Reading and plotting shapefile: {shapefile_path_base}")
#     try:
#         m.readshapefile(
#             shapefile_path_base,
#             'myshape',
#             drawbounds=True,
#             color='green', # Color of the shapefile lines
#             linewidth=0.8
#         )
#         print("Shapefile plotted successfully.")
#     except Exception as e:
#         print(f"ERROR: Could not read or plot shapefile: {e}")
#         print("Please check the path and ensure .shp, .shx, and .dbf files exist.")

#     print("Plotting lat/lon points...")
#     # Convert lat/lon points to map projection coordinates (x, y)
#     # For 'cyl' projection, this is often close to a direct mapping,
#     # but using the Basemap object ensures correctness.
#     x, y = m(lons, lats)

#     # Plot the points using scatter
#     m.scatter(
#         x,
#         y,
#         marker='o',      # Marker style
#         color='red',     # Marker color
#         s=40,            # Marker size
#         zorder=5         # Ensure points are on top
#     )
#     print("Points plotted.")

#     plt.title("Simple Shapefile and Points Plot")

#     print("Displaying plot...")
#     plt.show()
#     print("Done.")

# # plot_map("Shapefile/states_IND.shp", (74, 86.8), (8, 20.8))

def print_keys_and_metadata(fpath):
    with hf.File(fpath) as f:
        def print_metadata(name, obj):
            print(f"\nObject: {name}")
            obj_type = "Group" if isinstance(obj, hf.Group) else "Dataset"
            print(f"  Type: {obj_type}")
            print("  Attributes:")
    
            # Print attributes if available
            if obj.attrs:
                for attr_name, attr_value in obj.attrs.items():
                    if isinstance(attr_value, np.ndarray):
                        # Convert small arrays to list for better display
                        attr_value = attr_value.tolist() if len(attr_value) <= 10 else "[Array with large size]"
                    print(f"    - {attr_name}: {attr_value}")
            else:
                print("    No attributes available.")
    
        f.visititems(print_metadata)

fpath = "/home/sac/data_67/Nowcasting/data/required_ne/202102180000.HDF5"
with hf.File(fpath, 'r') as f:
    print(f.keys())