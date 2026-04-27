# %% [markdown]
# # Habitat suitability under climate change
# 
# Our changing climate is changing where plant species can live,
# and conservation and restoration practices will need to take
# this into
# account.
# 
# In this coding challenge, you will create a habitat suitability model
# for a terrestrial plant species of your choice that lives in the contiguous United States
# (CONUS). We have this limitation because the downscaled climate data we
# suggest, the [MACAv2 dataset](https://www.climatologylab.org/maca.html),
# is only available in the CONUS – if you find other downscaled climate
# data at an appropriate resolution, you are welcome to choose a different
# study area. If you don’t have anything in mind, you can take a look at
# [*Sorghastrum nutans*](https://www.gbif.org/species/2704414), a grass native to North America. In the past 50
# years, its range has moved
# northward.
# 
# Your suitability assessment will be based on combining multiple data
# layers related to soil, topography, and climate, then applying a fuzzy logic model across the different data layers to generate habitat suitability maps. 
# 
# You will need to create a **modular, reproducible, workflow** using functions and loops.
# To do this effectively, we recommend planning your code out in advance
# using a technique such as a pseudocode outline or a flow diagram. We
# recommend breaking each of the blocks below out into multiple steps. It
# is unnecessary to write a step for every line of code unless you find
# that useful. As a rule of thumb, aim for steps that cover the major
# structures of your code in 2-5 line chunks.

# %%
# Import the below libraries:
"""For faster loading, consider importing the libraries in separate cells."""
# To create reproducible file paths
import os # To interact with the operating system
from pathlib import Path # To create Path objects
import pathlib # To access the object-oriented library for paths

# For unzipping folders
import time # To handle time-related tasks
import zipfile # To download and extract zip files

# For saving Python objects
import pickle # To pickle datasets that take long periods to process

# To use APIs
import earthaccess # For logging into NASA Earth Access
import requests # For sending API requests

# To download Global Biodiversity Information Facility (GBIF) data
from getpass import getpass # To obtain a GBIF login and password
import pygbif.occurrences as occ # To download species occurrence data from GBIF
import pygbif.species as species # To identify specific species datasets to download on GBIF

# To work with different types of data
import geopandas as gpd # To make GeoDataFrames/work with vector data
from glob import glob # To combine data arrays
from math import floor, ceil # For dealing with integers
import numpy as np # To work with arrays
import pandas as pd # To work with dataframes
import rioxarray as rxr # To work with raster data
import rioxarray.merge as rxrm # To merge raster data
from shapely.geometry import MultiPolygon, Polygon # To handle invalid geometries
import xarray as xr # To use xarray datasets
import xrspatial # To handle spatial data

# For visualization and interactive plotting
import holoviews as hv # To create interactive hvPlots
import hvplot.pandas # To enable hvPlot interactive plotting for Pandas dataframes
import hvplot.xarray # To enable hvPlot interactive plotting for xarray datasets
import matplotlib.gridspec as gridspec # To display the plot legends properly
import matplotlib.pyplot as plt # To import the Pyplot module

# %%
# Create a designated folder for the repository data
data_dir = os.path.join(
    pathlib.Path.home(),
    # In the earth-analytics data folder
    'earth-analytics',
    'data',
    # Specify the destination as inside the "spring-03-habitat-suitability-climate-change-_____" repository
    'spring-03-habitat-suitability-climate-change-Livian-Von-Dran',
    'redwood_habitat_suitability'
)

# Create the directory
os.makedirs(data_dir, exist_ok=True)

# %% [markdown]
# ## STEP 1: Study overview
# 
# Before you begin coding, you will need to design your study.
# 
# ### Step 1a: Select a species
# Select the terrestrial plant species you want to study, and research its habitat parameters in scientific studies or other reliable sources. Individual studies may not have the breadth needed for this purpose, so take a look at reviews or overviews of the data. Do **not** just look at an AI-generated summary! In the US, the National Resource Conservation Service can have helpful fact sheets about different species. University Extension programs are also good resources for summaries.</p>
# <p>Based on your research, select soil, topographic, and climate variables that you can use to determine if a particular location and time period is a suitable habitat for your species.</p></div></div>
# 
# **Reflect and respond**: 
# Write a description of your species. What habitat is it found in? What is its geographic range? What, if any, are conservation threats to the species? What data will shed the most light on habitat suitability for this species? 
# 
# What core scientific question do you hope to answer about potential future changes in habitat suitability? Don't forget to cite your sources!

# %% [markdown]
# Your response here:

# %% [markdown]
# The species I have chosen is known by several names, chief among them being the California redwood or coast redwood (*Sequoia sempervirens*). Belonging to the Sequoioideae, this species is among the largest tree species ever documented, making it a desirable target for logging operations. Historically, redwood forests were restricted to a 900 kilometer belt along the Coast Range that spanned between central California and southern Oregon (Lorimer et al., 2009), but logging has removed 95% of the existing old-growth forests (Save the Redwoods League, 2018). Remaining old-growth redwood forests continue to be threatened by deforestation, invasive species, and increasingly destructive wildfires (Lorimer et al., 2009). Given that a previous project measured fog occurrence to support redwood habitat assessments, such data may be useful to determine habitat suitability for the California redwood (Wernet et al., 2020). For my core scientific question, I wish to address how altered temperature and precipitation will affect the habitat suitability of the already limited Calfiornia redwood range.
# 
# References
# 
# Burns, E. E., Campbell, R., & Cowan, P. D. (2018). *State of Redwoods Conservation Report*. https://www.savetheredwoods.org/wp-content/uploads/State-of-Redwoods-Conservation-Report-Final-web.pdf
# 
# Lorimer, C. G., Porter, D. J., Madej, M. A., Stuart, J. D., Veirs, S. D., Norman, S. P., O’Hara, K. L., & Libby, W. J. (2009). Presettlement and modern disturbance regimes in coast redwood forests: Implications for the conservation of old-growth stands. *Forest Ecology and Management*, *258*(7), 1038–1054. https://doi.org/10.1016/j.foreco.2009.07.008
# 
# Werner, Z., Berger, A., Winter, A., Choi, C. T. H., Evangelista, P., Jarnevich, C., Vorster, T., Woodward, B., & Young N. (2020). *California & Oregon Ecological Forecasting: Detecting and Forecasting Fog Occurrence, Frequency, and Change to Support Coast Redwood (Sequoia sempervirens) Habitat Assessments*. https://ntrs.nasa.gov/api/citations/20205011382/downloads/2020Fall_CO_California%26OregonEco_ProjectSummary_FD-final.docx.pdf
# 

# %%
# Create a directory for the GBIF data
gbif_dir = os.path.join(data_dir, 'gbif_redwood_dir')

# %%
# Permanently log into GBIF
# Do not reset credentials to avoid repeated login requests
reset_credentials = False

# Request and store username
if (not ('GBIF_USER'  in os.environ)) or reset_credentials:
    os.environ['GBIF_USER'] = input('GBIF username:')

# Request and store password
if (not ('GBIF_PWD'  in os.environ)) or reset_credentials:
    os.environ['GBIF_PWD'] = getpass('GBIF password:')
    
# Request and store email address
if (not ('GBIF_EMAIL'  in os.environ)) or reset_credentials:
    os.environ['GBIF_EMAIL'] = input('GBIF email:')

# %%
# Check that the login attempt was successful
'GBIF_PWD' in os.environ

# %%
# Set the species name
species_name = "Sequoia sempervirens"

# Pull the species info from GBIF
species_info = species.name_lookup(species_name, rank = 'Species')

# Grab the first result and print it
first_result = species_info['results'][0]
first_result

# %%
# Get the species key
species_key = first_result['nubKey']

# Check what the species key is
first_result['species'], species_key

# %%
# Create the file path
gbif_pattern = os.path.join(gbif_dir, '*.csv')

# Create a function to download the redwood occurrence data
if not glob(gbif_pattern):
    # Submit the query to GBIF
    gbif_query = occ.download([
        f"speciesKey = {species_key}",
        "hasCoordinate = True",
    ])
    # Only download the data once
    if not 'GBIF_DOWNLOAD_KEY' in os.environ:
        os.environ['GBIF_DOWNLOAD_KEY'] = gbif_query[0]
        download_key = os.environ['GBIF_DOWNLOAD_KEY']
        time.sleep(5)
    # Download the data
    download_info = occ.download_get(
        os.environ['GBIF_DOWNLOAD_KEY'],
        path = data_dir
    )
    # Unzip the file
    with zipfile.ZipFile(download_info['path']) as download_zip:
        download_zip.extractall(path = gbif_dir)

# Locate the CSV file path
gbif_path = glob(gbif_pattern)[0]

# %%
# Look at the download information
occ.download_meta("0070387-260226173443078") # Input the download key to view the information

# %%
# Read the CSV
gbif_df = pd.read_csv(
    gbif_path,
    delimiter = '\t'
)

# Check the dataframe
gbif_df

# %%
# Look at the columns
gbif_df.columns

# %%
# Convert the dataframe into a geodataframe (GDF)
gbif_gdf = (
    gpd.GeoDataFrame(
        gbif_df,
        # Add geometry columns to convert to a GDF
        geometry = gpd.points_from_xy(
            gbif_df.decimalLongitude,
            gbif_df.decimalLatitude
        ),
        crs = 'EPSG:4326'
    )
)

# Display the GDF data
gbif_gdf

# %%
# Create an interactive plot of the GDIF redwood observation data
gbif_gdf.hvplot(
    geo = True,
    tiles = 'EsriImagery',
    title = 'Redwood Observations on GDIF',
    # Avoid using a fill color, but select a line color of your choice
    fill_color = None,
    line_color = 'purple',
)

# %% [markdown]
# ### Step 1b: Select study sites
# Based on your research and/or range maps you find online, select at least 2 sites where your species occurs. These could be national parks, national forests, national grasslands or other protected areas, or some other area you're interested in. You can access protected area polygons from the [US Geological Survey's Protected Area Database](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-overview), [national grassland units](https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.NationalGrassland.zip), etc.
# 
# When selecting your sites, you might want to look for places that are marginally habitable for this species, since those locations will be most likely to show changes due to climate.
# 
# Generate a site map for each location.

# %%
# Create a directory for the park sites
site_dir = Path(data_dir) / "redwood_sites"
site_dir.mkdir(parents=True, exist_ok=True)

# Create a path to access the zip file
"""Most California redwoods are found in protected state and National Parks in California, so the database of choice is  
the California Protected Areas Database (CPAD). As of 2025, the name of the zip file is "cpad_release_2025b.zip,"
downloadable at 
https://data.cnra.ca.gov/dataset/0ae3cd9f-0612-4572-8862-9e9a1c41e659/resource/27323846-4000-42a2-85b3-93ae40edeff9/download/cpad_release_2025b.zip."""
zip_path = site_dir / "cpad_release_2025b.zip"

# Print the zip file path
print(zip_path)

# %%
# Establish the redwood site URL for the download
redwood_url = "https://data.cnra.ca.gov/dataset/0ae3cd9f-0612-4572-8862-9e9a1c41e659/resource/27323846-4000-42a2-85b3-93ae40edeff9/download/cpad_release_2025b.zip"

# Download the data only once
if not zip_path.exists():
     with requests.get(redwood_url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# Extract the files
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(site_dir)

# %%
# Define a path to the desired shapefile
"""Individual state and National Parks will be distinct layers in the units shapefile."""
redwood_shp = "redwood_habitat_suitability/redwood_sites/CPAD_Release_2025b/CPAD_2025b_Units/CPAD_2025b_Units.shp"

# Display the columns in the shapefile
cpad_units = gpd.read_file(redwood_shp)
cpad_units.head()

# %%
# Filter by "UNIT_NAME" for Redwood National Park and display the results
rnp_gdf = cpad_units[cpad_units["UNIT_NAME"] == "Redwood National Park"] 
rnp_gdf

# %%
# Filter by "UNIT_NAME" for Humboldt Redwoods State Park and display the results
hrsp_gdf = cpad_units[cpad_units["UNIT_NAME"] == "Humboldt Redwoods State Park"]
hrsp_gdf                                                                       

# %%
# Reproject both GDFs to ESPG:4326 for plotting
rnp_gdf_refined = rnp_gdf.to_crs("EPSG:4326")
hrsp_gdf_refined = hrsp_gdf.to_crs("EPSG:4326")

# %%
# Plot the Redwood National Park GDF
rnp_plot = rnp_gdf_refined.hvplot(
    fill_color = None, # Do not use a fill color
    frame_width = 600, # Set the plot width
    geo = True,
    line_color = "white", # Use white for the boundary line
    tiles = 'EsriImagery',
    title = 'Redwood National Park', # Set the title
)
rnp_plot

# %%
# Plot the Humboldt Redwoods State Park GDF
hrsp_plot = hrsp_gdf_refined.hvplot(
    fill_color = None, # Do not use a fill color
    frame_width = 600, # Set the plot width
    geo = True,
    line_color = "white", # Use white for the boundary line
    tiles = 'EsriImagery',
    title = 'Humboldt Redwoods State Park', # Set the title
)
hrsp_plot

# %% [markdown]
# **Reflect and Respond**: 
# Write a site description for each of your sites, or for all of your sites as a group if you have chosen a large number of linked sites. What
# differences or trends in habitat suitability over time do you expect to see among your sites?

# %% [markdown]
# Your response here:

# %% [markdown]
# Due to the limited geographic distribution of the California redwood, I was forced to pick two sites within California. To maximize site diversity, I chose Redwood National Park, a national park in northern California, and a Humboldt Redwoods National Park, a state park in north-central California. Both parks are only a degree apart latitude-wise, but I anticipate that its more northern latitude and better funding render Redwood National Park more prepared to weather the impacts of climate change.

# %% [markdown]
# ### Step 1c: Select time periods
# 
# In general when studying climate, we are interested in **climate
# normals**, which are typically calculated from 30 years of data so that
# they reflect the climate as a whole and not a single year which may be
# anomalous. So if you are interested in the climate around 2050, you will need to access climate data from 2035-2065.
# 
# **Reflect and Respond**: Select at least two 30-year time periods to compare, such as historical and 30 years into the future. These time periods should help you to answer your scientific question.

# %% [markdown]
# Your response here:

# %% [markdown]
# The two time periods I have selected are 1970-1999 and 2040-2069. These are time periods for which the Climate Futures Toolbox Future Climate Scatter tool has data.

# %% [markdown]
# ### Step 1d: Select climate models
# 
# There is a great deal of uncertainty among the many global climate
# models available. One way to work with the variety is by using an
# **ensemble** of models to try to capture that uncertainty. This also
# gives you an idea of the range of possible values you might expect! To
# be most efficient with your time and computing resources, you can use a
# subset of all the climate models available to you. However, for each
# scenario, you should attempt to include models that are:
# 
# -   Warm and wet
# -   Warm and dry
# -   Cold and wet
# -   Cold and dry
# 
# for each of your sites.
# 
# To figure out which climate models to use, you will need to access
# summary data near your sites for each of the climate models. You can do
# this using the [Climate Futures Toolbox Future Climate Scatter
# tool](https://climatetoolbox.org/tool/Future-Climate-Scatter). There is
# no need to write code to select your climate models, since this choice
# is something that requires your judgement and only needs to be done
# once.
# 
# If your question requires it, you can also choose to include multiple
# climate variables, such as temperature and precipitation, and/or
# multiple emissions scenarios, such as RCP4.5 and RCP8.5.
# 
# **Reflect and respond**: Choose at least 4 climate models that cover the range of possible future climate variability at your sites. Which models did you choose, and how did you make that decision?

# %% [markdown]
# Your response here (don't forget to cite the Climate Toolbox): 

# %% [markdown]
# For the four climate models, I chose CanESM2, MIROC-ESM-CHEM, MRI-CGCM3, and NorESM1-M for warm and wet, warm and dry, cooler and wet, and cooler and dry climate scenarios respectively. The Climate Futures Toolbox Future Climate Scatter tool was used to view future (2040-2069) temperature and precipitation projections for Redwood National Park and Humboldbt Redwoods State Park in the RCP 8.5 scenario; the models were matched to the climate scenario they best represented on these scatter plots (Hegewisch et al.). As California redwoods are reliant on fog cover to obtain hydration during the dry summer months, it is essential that some of these models account for the intensifying drought and warming temperates in California—conditions thought to have contributed an estimated 33% drop in fog since the 20th century (Johnstone & Dawson, 2010).
# 
# References
# 
# Hegewisch, K.C., Laquindanum, V., Fleishman, E., Hartmann, H., & Mills-Novoa, M. *Climate Toolbox*. https://climatetoolbox.org/tool/Future-Climate-Scatter
# 
# Johnstone, J. A., & Dawson, T. E. (2010). Climatic context and ecological implications of summer fog decline in the coast redwood region. *Proceedings of the National Academy of Sciences*, *107*(10), 4533–4538. https://doi.org/10.1073/pnas.0915062107

# %% [markdown]
# ## STEP 2: Data access
# 
# ### Step 2a: Soil data
# 
# The [POLARIS dataset](http://hydrology.cee.duke.edu/POLARIS/) is a
# convenient way to uniformly access a variety of soil parameters such as
# pH and percent clay in the US. It is available for a range of depths (in
# cm) and split into 1x1 degree tiles.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Write a <strong>function with a numpy-style docstring</strong> that
# will download POLARIS data for a particular location, soil parameter,
# and soil depth. Your function should account for the situation where
# your site boundary crosses over multiple tiles, and merge the necessary
# data together.</p>
# <p>Then, use loops to download and organize the rasters you will need to
# complete this section. Include soil parameters that will help you to
# answer your scientific question. We recommend using a soil depth that
# best corresponds with the rooting depth of your species.</p></div></div>

# %%
# Create a function to return loopable URLs for soil data from the POLARIS dataset
def create_polaris_urls(soil_prop, stat, soil_depth, gdf_bounds):
    """
    Variables:
    soil_prop: Desired soil property (ex. soil pH)
    stat: Statistic (ex. mean, median, maximum)
    soil_depth: Soil depth in cm
    gdf_bounds: Data array of site boundaries

    Output:
    soil_urls: A list of the soil URLs
    """
    # Extract the bounding box for the site
    xmin, ymin, xmax, ymax = gdf_bounds
    # Snap the boundary to whole degree
    min_lon = floor(xmin)
    max_lon = ceil(xmax)
    min_lat = floor(ymin)
    max_lat = ceil(ymax)
    # Generate a list of soil URLs
    soil_urls = []
    
    # Initiate a loop to obtain tiles
    for lon in range(min_lon, max_lon):
        for lat in range(min_lat, max_lat):
            # Define the tile corners
            current_max_lon = lon + 1
            current_max_lat = lat + 1
            # Define the URL template
            url_template = (
                "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/"
                # Insert the following args:
                # The wanted soil property
                "{soil_prop}/"
                # The statistic
                "{stat}/"
                # The soil depths
                "{soil_depth}/"
                # The GDF bounds
                "lat{min_lat}{max_lat}_lon{min_lon}{max_lon}.tif"
            )
            # Fill in the template
            soil_url_template = url_template.format(
                soil_prop = soil_prop,
                stat = stat,
                soil_depth = soil_depth,
                min_lat = lat, 
                max_lat = current_max_lat,
                min_lon = lon, 
                max_lon = current_max_lon
            )
            # Append URLs to the list
            soil_urls.append(soil_url_template)
        
    # Return all soil URLs
    return soil_urls

# %%
# Create a loop to pull multiple sites and properties:
# For pulling the selected redwood sites
sites = {
    'rnp': rnp_gdf_refined,
    'hrsp': hrsp_gdf_refined
}

# For pulling the soil pH and bulk density
soil_props = ['ph', 'bd']

# Create an empty dictionary
soil_urls = {}

# Loop through the sites
for site_name, gdf in sites.items():
    # Create a sub-dictionary for each site
    soil_urls[site_name] = {}
    # Loop through the soil properties
    for prop in soil_props:
        # Create URLs for each property
        soil_urls[site_name][prop] = create_polaris_urls(
            # Specify the soil property
            soil_prop=prop,
            # Obtain the average value
            stat='mean',
            # Restrict soil depth to 5-15 cm
            soil_depth='5_15',
            # Define the bounding box for a site
            gdf_bounds=gdf.total_bounds
        )

# %%
# List the obtained soil URLs
soil_urls

# %%
# Create a function that downloads data from the URLs and converts the data to a data array:
def build_da(urls, bounds):
    """
    Variables:
    urls: List of POLARIS URLs
    bounds: Redwood site boundaries

    Output:
    xarray.DataArray: Merged data array
    """
    # Generate a list of data arrays
    all_das = []
    # Create a buffer
    buffer = 0.025
    xmin, ymin, xmax, ymax = bounds
    bounds_buffer = (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)

    # Process the data arrays sequentially
    for url in urls:
        # Open the raster to mask missing data and remove extra dimensions
        tile_da = rxr.open_rasterio(url,
                                    mask_and_scale=True).squeeze()
        # Crop the tile buffered boundaries
        cropped_da  = tile_da.rio.clip_box(*bounds_buffer)
        # Store the cropped tile
        all_das.append(cropped_da)
    
    # Merge all data arrays into a single raster
    merged_raster = rxrm.merge_arrays(all_das)

    # Return the merged raster
    return merged_raster

# %%
# Create an empty dictionary
soil_das = {}

for site_name, gdf in sites.items():
    soil_das[site_name] = {}
    for prop, urls in soil_urls[site_name].items():
        if not urls:
            print(f"WARNING: No URLs for {prop} at {site_name}")
            continue
        try:
            soil_das[site_name][prop] = build_da(urls, gdf.total_bounds)
        except Exception as e:
            print(f"ERROR processing {prop} at {site_name}: {e}")

# %%
# List the combined data arrays
soil_das

# %%
# Create label variables to be inserted into the plot title and color bar
prop_labels = {'ph': 'Soil pH', 'bd': 'Bulk Density (g/cm³)'}

# Configure the color scheme for both plots
prop_cmaps = {'ph': 'coolwarm', 'bd': 'viridis'}

# %%
# Add labels for the elevation, slope, and aspect
topography_labels = {
   'elevation': 'Elevation (m)',
    'slope': 'Slope (°)',
    'aspect': 'Aspect (°)'
}

# Color code the labels
topography_cmaps = {
    'elevation': 'terrain',
    'slope': 'plasma',
    'aspect': 'twilight'
}

# Create a function to plot the elevations, slopes, and aspects of RNP and HRSP:
for topography_var in topography_labels:
    rnp_da = topography_das['rnp'][topography_var]
    hrsp_da = topography_das['hrsp'][topography_var]

    # Configure the color map for both sites
    cmap = topography_cmaps[topography_var]

    # Set minimum and maximum for the color scale
    vmin = min(float(rnp_da.min()), float(hrsp_da.min()))
    vmax = max(float(rnp_da.max()), float(hrsp_da.max()))

    # Specify the size of the figures
    fig = plt.figure(figsize=(14, 7))

     # Create the gridspec layout
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], hspace=0.4)

    # Set the axes
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, :])

    # Plot the elevation, slope, and aspect at Redwood National Park
    # Set the axis to axis 1 for the Redwood National Park plot
    rnp_da.plot(ax=ax1, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    # Reproject the GeoDataFrame to match the raster's CRS
    rnp_gdf_rp = rnp_gdf_refined.to_crs(rnp_da.rio.crs)
    # Plot the reprojected boundaries on top of the raster
    rnp_gdf_rp.plot(ax=ax1, facecolor='none', edgecolor='white', linewidth=1)
    # Set axes limits to the extent of the park boundaries
    xmin, ymin, xmax, ymax = rnp_gdf_rp.total_bounds
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    # Set the title
    ax1.set_title(f'{topography_labels[topography_var]} at Redwood National Park')

    # Plot the elevation, slope, and aspect at Humboldt Redwoods State Park
    # Set the axis to axis 2 for the Humboldt Redwoods State Park plot
    img = hrsp_da.plot(ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    # Reproject the GeoDataFrame to match the raster's CRS
    hrsp_gdf_rp = hrsp_gdf_refined.to_crs(hrsp_da.rio.crs)
    # Plot the reprojected boundaries on top of the raster
    hrsp_gdf_rp.plot(ax=ax2, facecolor='none', edgecolor='white', linewidth=1)
    # Set axes limits to the extent of the park boundaries
    xmin, ymin, xmax, ymax = hrsp_gdf_rp.total_bounds
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    # Set the title
    ax2.set_title(f'{topography_labels[topography_var]} at Humboldt Redwoods State Park')

    # Set the color bar
    fig.colorbar(img, cax=cax, orientation='horizontal', label=topography_labels[topography_var])

    # Map the plots
    plt.show()

# %%
# Create label variables to be inserted into the plot title and color bar
prop_labels = {'ph': 'Soil pH', 'bd': 'Bulk Density (g/cm³)'}

# Configure the color scheme for both plots
prop_cmaps = {'ph': 'coolwarm', 'bd': 'viridis'}

# Create a function that visualizes the soil pH and bulk density data for both sites:
# Loop over the soil properties
for prop in soil_props:
 # Extract the raster for both sites
    rnp_da = soil_das['rnp'][prop]
    hrsp_da = soil_das['hrsp'][prop]
    # Configure the color map
    cmap = prop_cmaps[prop]
    # Set minimum and maximum for the color scale
    vmin = float(min(rnp_da.min(), hrsp_da.min()))
    vmax = float(max(rnp_da.max(), hrsp_da.max()))
    # Create the gridspec layout
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], hspace=0.4)
    # Specify the figure size
    fig = plt.figure(figsize=(14, 7))
    # Create a dedicated axes object for the horizontal color bar
    cax = fig.add_subplot(gs[1, :])
    # Set the axes for both plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Plot the soil pH and bulk density at Redwood National Park
    # Set the axis to axis 1 for the Redwood National Park plot
    rnp_da.plot(ax=ax1, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    # Reproject the GeoDataFrame to match the raster's CRS
    rnp_gdf_rp = rnp_gdf_refined.to_crs(rnp_da.rio.crs)
    # Plot the reprojected boundaries on top of the raster
    rnp_gdf_rp.plot(ax=ax1, facecolor='none', edgecolor='white', linewidth=1)
    # Set axes limits to the extent of the park boundaries
    xmin, ymin, xmax, ymax = rnp_gdf_rp.total_bounds
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    # Add axes labels for longitude and latitude
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    # Set the title with the property label and reduce the font size to avoid title overlap
    ax1.set_title(f'{prop_labels[prop]} of 5-15 cm Deep Soil in Redwood National Park\n', fontsize=10)

    # Plot the elevation, slope, and aspect at Humboldt Redwoods State Park
    # Set the axis to axis 2 for the Humboldt Redwoods State Park plot
    hrsp_soil = hrsp_da.plot(ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    # Reproject the GeoDataFrame to match the raster's CRS
    hrsp_gdf_rp = hrsp_gdf_refined.to_crs(hrsp_da.rio.crs)
    # Plot the reprojected boundaries on top of the raster
    hrsp_gdf_rp.plot(ax=ax2, facecolor='none', edgecolor='white', linewidth=1)
    # Set axes limits to the extent of the park boundaries
    xmin, ymin, xmax, ymax = hrsp_gdf_rp.total_bounds
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    # Add axes labels for longitude and latitude
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    # Set the title with the property label and reduce the font size to avoid title overlap
    ax2.set_title(f'{prop_labels[prop]} of 5-15 cm Deep Soil in Humboldt Redwoods State Park\n', fontsize=10)

    # Set the color bar
    fig.colorbar(hrsp_soil, cax=cax, orientation='horizontal', label=prop_labels[prop])

    # Map the plots
    plt.show()

# %% [markdown]
# ### Step 2b: Topographic data
# 
# Depending on your species habitat needs/environmental parameters, you might be interested in elevation, slope, and/or aspect. You can access reliable elevation data from the [SRTM
# dataset](https://www.earthdata.nasa.gov/data/instruments/srtm),
# available through the [earthaccess
# API](https://earthaccess.readthedocs.io/en/latest/quick-start/). Once you have elevation data, you can calculate slope and aspect.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Write a <strong>function with a numpy-style docstring</strong> that
# will download SRTM elevation data for a particular location and
# calculate any additional topographic variables you need such as slope or
# aspect.</p>
# <p>Then, use loops to download and organize the rasters you will need to
# complete this section. Include topographic parameters that will help you
# to answer your scientific question.</p></div></div>
# 
# > **Warning**
# >
# > Be careful when computing the slope from elevation that the units of
# > elevation match the projection units (e.g. meters and meters, not
# > meters and degrees). You will need to project the SRTM data to
# > complete this calculation correctly.

# %%
# Create a directory for the topography data
topography_dir = os.path.join(data_dir, "topography_datasets")
os.makedirs(topography_dir, exist_ok=True)

# %%
# Login to NASA Earth Access
earthaccess.login()

# %%
# Access the SRTM data using the keyword "SRTM DEM"
datasets = earthaccess.search_datasets(keyword = "SRTM DEM")

# Print the accessed datasets
for dataset in datasets:
    print(dataset['umm']['ShortName'], dataset['umm']['EntryTitle'])

# %%
# Create a function that can download SRTM elevation data and derive slopes and aspect rasters for a site:
def topography_data(site_name, site_gdf, topography_dir):
    """""
    Variables:
    site_name: An abbreviation for the site name
    site_gdf: The site boundary GeoDataFrame
    topography_dir: A directory to store downloaded topography data
    """
    # Create a new directory that joins the topography directory and the site name
    site_topography_dir = os.path.join(topography_dir, site_name)
    os.makedirs(site_topography_dir, exist_ok=True)
    # Set the SRTM pattern
    srtm_pattern = os.path.join(site_topography_dir, "*.hgt.zip")
    # Establish the buffer boundary
    buffer = 0.025
    xmin, ymin, xmax, ymax = tuple(site_gdf.total_bounds)
    bounds_buffer = (xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)

    # Proceed with the download if the globbed pattern is not already present
    if not glob(srtm_pattern):
        srtm_search = earthaccess.search_data(
            short_name='SRTMGL3',
            bounding_box=bounds_buffer
        )
        earthaccess.download(srtm_search, site_topography_dir)
    
    # Generate a list of data arrays
    da_list = []
    # Open and merge tiles
    for srtm_path in glob(srtm_pattern):
        tile_da = rxr.open_rasterio(srtm_path, mask_and_scale=True).squeeze()
        da_list.append(tile_da.rio.clip_box(*bounds_buffer))
    # Define the topography data array
    elevation_da = rxrm.merge_arrays(da_list)
    # Aspect the raster
    aspect_da = xrspatial.aspect(elevation_da)
    # Reproject the elevation data array
    elevation_da_rp = elevation_da.rio.reproject('EPSG:5070')
    # Derive the slope from the dataset
    slope_da = xrspatial.slope(elevation_da_rp).rio.reproject('EPSG:4326')

    # Return the data arrays for the elevation, slope, and aspect
    return {'elevation': elevation_da, 'slope': slope_da, 'aspect': aspect_da}

# %%
# Create an empty dictionary
topography_das = {}

# Loop over the sites cdictionary
# Call "topography_data"
for site_name, gdf in sites.items():
    topography_das[site_name] = topography_data(site_name, gdf, topography_dir)

# %%
# List the topography data arrays
topography_das

# %%
# Add labels for the elevation, slope, and aspect
topography_labels = {
   'elevation': 'Elevation (m)',
    'slope': 'Slope (°)',
    'aspect': 'Aspect (°)'
}

# Color code the labels
topography_cmaps = {
    'elevation': 'terrain',
    'slope': 'plasma',
    'aspect': 'twilight'
}

# Create a function to plot the elevations, slopes, and aspects of RNP and HRSP:
for topography_var in topography_labels:
    rnp_da = topography_das['rnp'][topography_var]
    hrsp_da = topography_das['hrsp'][topography_var]

    # Configure the color map for both sites
    cmap = topography_cmaps[topography_var]

    # Set minimum and maximum for the color scale
    vmin = min(float(rnp_da.min()), float(hrsp_da.min()))
    vmax = max(float(rnp_da.max()), float(hrsp_da.max()))

    # Specify the size of the figures
    fig = plt.figure(figsize=(14, 7))

     # Create the gridspec layout
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], hspace=0.4)

    # Set the axes
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, :])

    # Plot the elevation, slope, and aspect at Redwood National Park
    # Set the axis to axis 1 for the Redwood National Park plot
    rnp_da.plot(ax=ax1, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    # Reproject the GeoDataFrame to match the raster's CRS
    rnp_gdf_rp = rnp_gdf_refined.to_crs(rnp_da.rio.crs)
    # Plot the reprojected boundaries on top of the raster
    rnp_gdf_rp.plot(ax=ax1, facecolor='none', edgecolor='white', linewidth=1)
    # Set axes limits to the extent of the park boundaries
    xmin, ymin, xmax, ymax = rnp_gdf_rp.total_bounds
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    # Set the title
    ax1.set_title(f'{topography_labels[topography_var]} at Redwood National Park')

    # Plot the elevation, slope, and aspect at Humboldt Redwoods State Park
    # Set the axis to axis 2 for the Humboldt Redwoods State Park plot
    img = hrsp_da.plot(ax=ax2, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    # Reproject the GeoDataFrame to match the raster's CRS
    hrsp_gdf_rp = hrsp_gdf_refined.to_crs(hrsp_da.rio.crs)
    # Plot the reprojected boundaries on top of the raster
    hrsp_gdf_rp.plot(ax=ax2, facecolor='none', edgecolor='white', linewidth=1)
    # Set axes limits to the extent of the park boundaries
    xmin, ymin, xmax, ymax = hrsp_gdf_rp.total_bounds
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    # Set the title
    ax2.set_title(f'{topography_labels[topography_var]} at Humboldt Redwoods State Park')

    # Set the color bar
    fig.colorbar(img, cax=cax, orientation='horizontal', label=topography_labels[topography_var])

    # Map the plots
    plt.show()

# %% [markdown]
# ### Step 2c: Climate model data
# 
# You can use MACAv2 data for historical and future climate data. Be sure
# to compare at least two 30-year time periods (e.g. historical vs. 10
# years in the future) for at least four of the CMIP models. Overall, you
# should be downloading at least 8 climate rasters for each of your sites,
# for a total of 16. **You will *need* to use loops and/or functions to do
# this cleanly!**.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Write a <strong>function with a numpy-style docstring</strong> that
# will download MACAv2 data for a particular climate model, emissions
# scenario, spatial domain, and time frame. Then, use loops to download
# and organize the 16+ rasters you will need to complete this section. The
# <a
# href="http://thredds.northwestknowledge.net:8080/thredds/reacch_climate_CMIP5_macav2_catalog2.html">MACAv2
# dataset is accessible from their Thredds server</a>. Include an
# arrangement of sites, models, emissions scenarios, and time periods that
# will help you to answer your scientific question.</p></div></div>

# %%
# Create a directory for the climate data
climate_dir = os.path.join(data_dir, 'climate_dir')
os.makedirs(climate_dir, exist_ok=True)

# Define the pattern and print it
climate_pattern = os.path.join(climate_dir, ".nc")
climate_pattern

# Define the path to save pickled climate files
pickle_file = os.path.join(climate_dir)

# %%
# Create a function that will convert Kelvin to Celsius in an xarray data array
def convert_temperature(temp):
    return temp - 273.15

# Create a function that will convert 0-360 longitude values to a -180 to 180 range
"""The longitude values will also be in an xarray data array."""
def convert_longitude(lon):
    return (lon -360) if lon > 180 else lon

# Create a function that will convert mm/month to mm/year for precipitation
def convert_precipitation(pr):
    return pr * 12


# %%
# Create a function that produces a date list range from the MACA data
def MACA_date_ranges(start_year, end_year):
    """
    Variables: 
    start_year: The earliest year in the dataset
    end_year: The final year in the dataset
    """
    # Generate a list of date range intervals
    intervals = []
    # Construct historical intervals in 5-year blocks
    intervals += [(y, y + 4) for y in range(1970, 2000, 5)]
    # Add a transitional interval
    intervals.append((2005, 2005))
    # Construct future intervals in 5-year blocks
    intervals += [(y, y + 4) for y in range(2040, 2070, 5)]
    # Return intervals only in the requested time range
    return [
        f"{s}_{e}"
        for s, e in intervals
        if s <= end_year and e >= start_year # Only keep intervals that overlap with "start_year" and "end_year"
    ]

# %%
# Specify the historical date range and print it
historical_date_range = MACA_date_ranges(1970, 1999) 
historical_date_range

# %%
# Specify the future date range and print it
future_date_range = MACA_date_ranges(2040, 2070)
future_date_range

# %%
# Combine the historical and future date ranges
date_ranges = historical_date_range + future_date_range
date_ranges

# %%
# Create a function that processes the MACA data
"""
Variables:
site_list: The combined list of Redwood National Park and Humboldt Redwoods State Park
date_ranges: The timeframe spanning both the historical (1980-2009) and future (2040-2069) data ranges
models: The four climate models
rcp_value: The RCP emissions scenario
variable_names: The list of the climate variable names
climate_dir: The directory to store the climate variable data

Output:
maca_results: A list of the processed MACA data
"""
def process_MACA_data(site_list, 
                      date_ranges, 
                      models,
                      rcp_value,
                      variable_names, 
                      climate_dir):
    
    # Generate a list of results
    maca_results = []

    # Loop over each site
    for site_name, site_gdf in site_list.items():
        # Reproject the shapefile's CRS to EPSG:4326 to match the raster's CRS
        site_rp = site_gdf.to_crs('EPSG:4326')
        xmin, ymin, xmax, ymax = site_rp.total_bounds
        # Add a buffer around the site bounds
        buffer = 0.025
        xmin, ymin, xmax, ymax = xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer

        # Create a loop for each date
        for date_range in date_ranges:
            start_year, end_year = map(int, date_range.split('_'))

            # Create a loop for each model
            for model in models:

                # Create a loop for each climate variable
                for variable_name in variable_names:
                    # Select the RCP value based on the data range
                    rcp_scenario = 'historical' if start_year <= 2005 else rcp_value
                    source_period = '1950_2005' if rcp_scenario == 'historical' else '2006_2099'
                    # Define the local path
                    maca_path = os.path.join(
                        climate_dir, 
                        f'maca_{model}_{site_name}_{variable_name}_{rcp_scenario}_{date_range}_CONUS_monthly.nc'
                    )
                    # Specify the MACA URL
                    maca_url = (
                        'https://tds-proxy.nkn.uidaho.edu/thredds/dodsC/'
                        f'agg_macav2metdata_{variable_name}_{model}_r1i1p1_'
                        f'{rcp_scenario}_{source_period}_CONUS_monthly.nc'
                    )

                    # If a file exists locally, load it
                    try:
                        if os.path.exists(maca_path):
                            maca_ds = xr.open_dataset(maca_path).squeeze()
                        # Otherwise, load from the URL
                        else:
                            maca_ds = xr.open_dataset(maca_url).squeeze()
                            maca_ds = maca_ds.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))

                        # Define the MACA data array
                        maca_da = maca_ds[list(maca_ds.data_vars)[0]]
                        # Convert the longitude range from 0-360 to -180 and 180
                        maca_da = maca_da.assign_coords(
                            lon=("lon", [convert_longitude(l) for l in maca_da.lon.values])
                        )
                        # Set the spatial dimensions
                        maca_da = maca_da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
                        # Clip the data to the site bounds
                        cropped_maca_da = maca_da.rio.clip_box(
                            minx=xmin, miny=ymin, maxx=xmax, maxy=ymax
                        )

                        # Check if the cropped data is empty
                        if cropped_maca_da.isnull().all():
                            print(f"No data for {site_name} at {date_range} with model {model} and variable {variable_name}")
                            continue

                        # Convert the temperature from Kelvin to Celsius for the temperature variables
                        if variable_name in ['tasmin', 'tasmax']:
                            cropped_maca_da = convert_temperature(cropped_maca_da)

                        # Convert the precipitation unit from mm/month to mm/year
                        if variable_name in ['pr']:
                            cropped_maca_da = convert_precipitation(cropped_maca_da)

                        # Append the results
                        maca_results.append(dict(
                            site_name=site_name,
                            climate_model=model,
                            climate_variable=variable_name,
                            date_range=date_range,
                            rcp=rcp_scenario,
                            da=cropped_maca_da
                        ))

                    # With the exception of OSErrors, skip missing datasets
                    except OSError:
                        print(f"Skipping missing dataset: {maca_url}")
                        continue

                    # Skip missing datasets and continue looping
                    except Exception as e:
                        print(f"Skipping dataset {maca_url} due to error: {e}")
                        continue

    # Return the generated results list
    return maca_results

# %%
# Specify the climate models used
"""Both sites will use the same four climate models."""
models = [
    # Warm and wet
    "CanESM2", 
    # Warm and dry
    "MIROC-ESM-CHEM", 
    # Cooler and wet
    "MRI-CGCM3", 
    # Cooler and dry
    "NorESM1-M"]

# Define the variables
variable_names = [
    # Precipitation
    'pr', 
    # Minimum temperature
    'tasmin', 
    # Maximum temperature
    'tasmax']

# Define the RCP scenario
rcp_value = "rcp85" # RCP 8.5 = high emissions climate scenario

# %%
# Define the path to pickle maca_output in the climate directory
maca_pickle = os.path.join(climate_dir, 'maca_output.pkl')

# Check if the pickle file exists in climate_dir
if os.path.exists(maca_pickle):
    # If it exists, load the pickled data
    with open(maca_pickle, 'rb') as f:
        maca_output = pickle.load(f)
    print(f"maca_output was loaded from '{maca_pickle}'.")
else:
    # If it doesn't exist, compute maca_output
    maca_output = process_MACA_data(
        site_list=sites,
        date_ranges=date_ranges,
        models=models,
        variable_names=variable_names,
        rcp_value=rcp_value,
        climate_dir=climate_dir
    )
    # Save the computed maca_output to climate_dir
    with open(maca_pickle, 'wb') as f:
        pickle.dump(maca_output, f)
    print(f"maca_output has been saved as '{maca_pickle}'.")

# %%
# Define the file paths for the pickled outputs within climate_dir
rnp_historical_pickle = os.path.join(climate_dir, 'rnp_historical_climate.pkl')
rnp_future_pickle = os.path.join(climate_dir, 'rnp_future_climate.pkl')
hrsp_historical_pickle = os.path.join(climate_dir, 'hrsp_historical_climate.pkl')
hrsp_future_pickle = os.path.join(climate_dir, 'hrsp_future_climate.pkl')

# If the data arrays are not pickled, obtain them
# Wrap each GDF in a dictionary
# For historical climate variables in Redwood National Park
if os.path.exists(rnp_historical_pickle):
    with open(rnp_historical_pickle, 'rb') as f:
        rnp_historical_climate = pickle.load(f)
    print(f"rnp_historical_climate was loaded from '{rnp_historical_pickle}'.")
else:
    rnp_historical_climate = process_MACA_data(
        {"rnp": rnp_gdf_refined},  # Wrapping rnp_gdf in a dictionary
        historical_date_range, 
        models,
        "historical", 
        ["pr", "tasmin", "tasmax"],  # Include all climate variables
        climate_dir)
    with open(rnp_historical_pickle, 'wb') as f:
        pickle.dump(rnp_historical_climate, f)
    print(f"rnp_historical_climate has been saved as '{rnp_historical_pickle}'.")

# For future predicted climate variables in Redwood National Park
if os.path.exists(rnp_future_pickle):
    with open(rnp_future_pickle, 'rb') as f:
        rnp_future_climate = pickle.load(f)
    print(f"rnp_future_climate was loaded from '{rnp_future_pickle}'.")
else:
    rnp_future_climate = process_MACA_data(
        {"rnp": rnp_gdf_refined}, 
        future_date_range, 
        models,
        "rcp85",  # RCP 8.5 = high emissions climate scenario
        ["pr", "tasmin", "tasmax"],
        climate_dir)
    with open(rnp_future_pickle, 'wb') as f:
        pickle.dump(rnp_future_climate, f)
    print(f"rnp_future_climate has been saved as '{rnp_future_pickle}'.")

# For historical climate variables in Humboldt Redwoods State Park
if os.path.exists(hrsp_historical_pickle):
    with open(hrsp_historical_pickle, 'rb') as f:
        hrsp_historical_climate = pickle.load(f)
    print(f"hrsp_historical_climate was loaded from '{hrsp_historical_pickle}'.")
else:
    hrsp_historical_climate = process_MACA_data(
        {"hrsp": hrsp_gdf},  # Wrapping hrsp_gdf in a dictionary
        historical_date_range, 
        models,
        "historical", 
        ["pr", "tasmin", "tasmax"],
        climate_dir)
    with open(hrsp_historical_pickle, 'wb') as f:
        pickle.dump(hrsp_historical_climate, f)
    print(f"hrsp_historical_climate has been saved as '{hrsp_historical_pickle}'.")

# For future predicted climate variables in Humboldt Redwoods State Park
if os.path.exists(hrsp_future_pickle):
    with open(hrsp_future_pickle, 'rb') as f:
        hrsp_future_climate = pickle.load(f)
    print(f"hrsp_future_climate was loaded from '{hrsp_future_pickle}'.")
else:
    hrsp_future_climate = process_MACA_data(
        {"hrsp": hrsp_gdf}, 
        future_date_range, 
        models,
        "rcp85", 
        ["pr", "tasmin", "tasmax"],
        climate_dir)
    with open(hrsp_future_pickle, 'wb') as f:
        pickle.dump(hrsp_future_climate, f)
    print(f"hrsp_future_climate has been saved as '{hrsp_future_pickle}'.")

# %%
# Create a function that obtains the average historical and future climate variables for both sites while incorporating the climate models:
"""
Variables:
historical_climate: List containing the averaged climate variables for the historical date range
future_climate: List containing the averaged climate variables for the historical date range

Output:
average_climate_period_results: A list of the fully averaged historical and future rasters
"""
def average_climate(historical_climate, future_climate):
    # Combine both lists into one to process all data in a single loop
    average_climate_list = historical_climate + future_climate
    # Generate the averaged results of all variables
    average_climate_results = []

    # Loop through each dataset
    for item in average_climate_list:
        # Pull out the data array and extract the raster data
        average_climate_da = item["da"]
        # Compute the average
        average_climate_raster = (
        average_climate_da.mean(dim="time")  # Collapse multiple months and years
    )
        # Use the RCP scenario to divide the dataset based on the date range
        scenario = "Historical" if item["rcp"] == "historical" else "Future"
        # Create a dictionary summarizing the dataset and add it to the list
        average_climate_results.append({
            "site_name": item["site_name"],
            "climate_model": item["climate_model"],
            "climate_variable": item["climate_variable"],
            "scenario": scenario,
            "date_range": item["date_range"],
            "average_climate_raster": average_climate_raster
            })
        
    # Create a list to hold the fully averaged historical and future rasters
    average_climate_period_results = []

    # Identify each unique site, model, variable, and scenario combination
    grouped_keys = {
        (
            item["site_name"],
            item["climate_model"],
            item["climate_variable"],
            item["scenario"]
        )
        for item in average_climate_results
    }

    # Loop through each unique grouping
    for site_name, climate_model, climate_variable, scenario in grouped_keys:
        # Collect all matching rasters for each identified grouping
        matching_rasters = [
            item["average_climate_raster"]
            for item in average_climate_results
            if item["site_name"] == site_name
            and item["climate_model"] == climate_model
            and item["climate_variable"] == climate_variable
            and item["scenario"] == scenario
        ]
        # Average all date-range chunks into one full raster
        average_climate_period_raster = xr.concat(
            matching_rasters,
            dim="date_range"
        ).mean(dim="date_range")
        # Append and store the fully averaged climate raster
        average_climate_period_results.append({
            "site_name": site_name,
            "climate_model": climate_model,
            "climate_variable": climate_variable,
            "scenario": scenario,
            "average_climate_raster": average_climate_period_raster
            })

    # Return the average climate results
    return average_climate_period_results 

# %%
# Obtain the averaged 30-year climate rasters for Redwood National Park and print them
rnp_average_climate_results = average_climate(
    historical_climate=rnp_historical_climate,
    future_climate=rnp_future_climate
)
rnp_average_climate_results

# %%
# Obtain the averaged 30-year climate rasters for Humboldt Redwoods State Park and print them
hrsp_average_climate_results = average_climate(
    historical_climate=hrsp_historical_climate,
    future_climate=hrsp_future_climate
)
hrsp_average_climate_results

# %%
# Create climate label variables to be inserted into the plot title and color bar
climate_prop_labels = {
    'pr': 'Average Annual Precipitation (mm)',
    'tasmin': 'Average Minimum Temperature (°C)',
    'tasmax': 'Average Maximum Temperature (°C)'
}

# Configure the color scheme for all plots
climate_prop_cmaps = {
    'pr': 'Blues',
    'tasmin': 'RdBu_r',
    'tasmax': 'RdBu_r'
}

# Create time label variables to be inserted into the plot title
time_prop_labels = {
    'Historical': '1970-1999',
    'Future': '2040-2069'
}

# %%
# Create a function that visualizes and plots the averaged climate rasters for Redwood National Park
for item in rnp_average_climate_results:
    # Extract the averaged climate raster
    rnp_da = item["average_climate_raster"]
    
    # Extract the climate variable and scenario
    prop = item["climate_variable"]
    scenario = item["scenario"]
    model = item["climate_model"]

    # Configure the color map
    cmap = climate_prop_cmaps[prop]

    # Set a minimum and maximum for the color scale
    vmin = float(rnp_da.min(skipna=True))
    vmax = float(rnp_da.max(skipna=True))

    # Create the gridspec layout
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], hspace=0.4)

    # Specify the figure size
    fig = plt.figure(figsize=(14, 7))

    # Create a subplot for the raster map
    ax1 = fig.add_subplot(gs[0, 0])

    # Create a dedicated axes object for the horizontal color bar
    cax = fig.add_subplot(gs[1, :])

    # Plot the raster without the built-in color bar
    rnp_plot = rnp_da.plot(
        ax=ax1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )

    # Assign the CRS if it is missing
    if rnp_da.rio.crs is None:
        rnp_da = rnp_da.rio.write_crs("EPSG:4326")

    # Reproject the GeoDataFrame to match the raster's CRS
    rnp_gdf_rp = rnp_gdf_refined.to_crs(rnp_da.rio.crs)

    # Plot the reprojected boundaries on top of the raster
    rnp_gdf_rp.plot(ax=ax1, facecolor='none', edgecolor='white', linewidth=1)

    # Set axes limits to the extent of the park boundaries
    xmin, ymin, xmax, ymax = rnp_gdf_rp.total_bounds
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    # Add axes labels for longitude and latitude
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # Set the title
    ax1.set_title(
    f'({model}) {climate_prop_labels[prop]} in Redwoods National Park '
    f'Between {time_prop_labels[scenario]} '
    f'in a High Emissions (RCP 8.5) Scenario'
    )

    # Add a horizontal color bar using the dedicated axes
    plt.colorbar(
        rnp_plot,
        cax=cax,
        orientation='horizontal',
        label=climate_prop_labels[prop]
    )

    # Display the plot
    plt.show()

# %%
# Create a function that visualizes and plots the averaged climate rasters for Humboldt Redwoods State Park
for item in hrsp_average_climate_results:
    # Extract the averaged climate raster
    hrsp_da = item["average_climate_raster"]
    
    # Extract the climate variable and scenario
    prop = item["climate_variable"]
    scenario = item["scenario"]
    model = item["climate_model"]

    # Configure the color map
    cmap = climate_prop_cmaps[prop]

    # Set a minimum and maximum for the color scale
    vmin = float(hrsp_da.min(skipna=True))
    vmax = float(hrsp_da.max(skipna=True))

    # Create the gridspec layout
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], hspace=0.4)

    # Specify the figure size
    fig = plt.figure(figsize=(14, 7))

    # Create a subplot for the raster map
    ax1 = fig.add_subplot(gs[0, 0])

    # Create a dedicated axes object for the horizontal color bar
    cax = fig.add_subplot(gs[1, :])

    # Plot the raster without the built-in color bar
    hrsp_plot = hrsp_da.plot(
        ax=ax1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )

    # Assign the CRS if it is missing
    if hrsp_da.rio.crs is None:
        hrsp_da = hrsp_da.rio.write_crs("EPSG:4326")

    # Reproject the GeoDataFrame to match the raster's CRS
    hrsp_gdf_rp = hrsp_gdf_refined.to_crs(hrsp_da.rio.crs)

    # Plot the reprojected boundaries on top of the raster
    hrsp_gdf_rp.plot(ax=ax1, facecolor='none', edgecolor='white', linewidth=1)

    # Set axes limits to the extent of the park boundaries
    xmin, ymin, xmax, ymax = hrsp_gdf_rp.total_bounds
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    # Add axes labels for longitude and latitude
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # Set the title
    ax1.set_title(
        f'({model}) {climate_prop_labels[prop]} in Humboldt Redwoods State Park '
        f'Between {time_prop_labels[scenario]} '
        f'in a High Emissions (RCP 8.5) Scenario '
    )

    # Add a horizontal color bar using the dedicated axes
    plt.colorbar(
        hrsp_plot,
        cax=cax,
        orientation='horizontal',
        label=climate_prop_labels[prop]
    )

    # Display the plot
    plt.show()

# %% [markdown]
# **Reflect and respond**: Make sure to include a description of the climate data and how you selected your models. Include a citation of the MACAv2 data.

# %% [markdown]
# Your response here:

# %% [markdown]
# The plots show a general trend of increasing average minimum and maximum temperatures and precipitation over time; the latter trend appears to conflict with predications that anthropogenic climate change will intensify drought in California (Diffenbaugh et al., 2015). The climate models were selected to cover the potential for climate change to intensify storms, drought, and rising temperatures, but only CRM-CRNM-5 and MIROC5 were able to be plotted successfully; greater precipitation is expected with CNRM-CM5 as it was selected to cover warm and wet scenarios, but MIROC5 also predicts more precipitation despite simulating cooler and dry scenarios (Joyce et al., 2018).
# 
# References
# 
# Diffenbaugh, N. S., Swain, D. L., & Touma, D. (2015). Anthropogenic warming has increased drought risk in California. *Proceedings of the National Academy of Sciences*, *112*(13), 3931–3936. https://doi.org/10.1073/pnas.1422385112
# 
# Joyce, L. A., Abatzoglou, J. T., & Coulson, D. P.. (2018). *Climate data for RPA 2020 Assessment: MACAv2 (METDATA) historical modeled (1950-2005) and future (2006-2099) projections for the conterminous United States at the 1/24 degree grid scale*. Forest Service Research Data Archive. https://doi.org/10.2737/rds-2018-0014

# %% [markdown]
# ## STEP 3: Harmonize data
# To use all your environmental and climate data layers together, you need to harmonize the different rasters you've downloaded and processed. 
# 
# As a first step, make sure that the grids for all the rasters match each other. Check out the <a href="https://corteva.github.io/rioxarray/stable/examples/reproject_match.html#Reproject-Match"><code>ds.rio.reproject_match()</code> method</a> from <code>rioxarray</code>. Make sure to use the data source that has the highest resolution as a template!</p></div></div>
# 
# > **Warning**
# >
# > If you are reprojecting data (as you need to here), the order of
# > operations is important! Recall that reprojecting will typically tilt
# > your data, leaving narrow sections of the data at the edge blank.
# > However, to reproject efficiently it is best for the raster to be as
# > small as possible before performing the operation. We recommend the
# > following process:
# >
# >     1. Crop the data, leaving a buffer around the final boundary
# >     2. Reproject to match the template grid (this will also crop any leftovers off the image)

# %%
# Select the highest resolution data source for both sites
"""For both datasets, the soil pH raster has the highest resolution."""
rnp_template_da = soil_das["rnp"]["ph"]
hrsp_template_da = soil_das["hrsp"]["ph"]

# %%
# Prepare the climate rasters for both sites
aligned_rnp_climate = [
    {
        **item,
        "average_climate_raster": (
            item["average_climate_raster"]
            .rio.write_crs("EPSG:4326")
            .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            .rio.reproject_match(rnp_template_da)
        )
    }
    for item in rnp_average_climate_results
]

aligned_hrsp_climate = [
    {
        **item,
        "average_climate_raster": (
            item["average_climate_raster"]
            .rio.write_crs("EPSG:4326")
            .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            .rio.reproject_match(hrsp_template_da)
        )
    }
    for item in hrsp_average_climate_results
]

# %%
# Prepare the soil rasters for both sites
aligned_rnp_soil = {
    prop: da.rio.reproject_match(rnp_template_da)
    for prop, da in soil_das["rnp"].items()
}

aligned_hrsp_soil = {
    prop: da.rio.reproject_match(hrsp_template_da)
    for prop, da in soil_das["hrsp"].items()
}

# %%
# Prepare the topography rasters for both sites
aligned_rnp_topography = {
    prop: da.rio.reproject_match(rnp_template_da)
    for prop, da in topography_das["rnp"].items()
}

aligned_hrsp_topography = {
    prop: da.rio.reproject_match(hrsp_template_da)
    for prop, da in topography_das["hrsp"].items()
}

# %%
# Print the template and boundary bounds of both sites to ensure their longitude and latitude parameters are closely aligned
print("HRSP template bounds:", hrsp_template_da.rio.bounds())
print("HRSP boundary bounds:", hrsp_gdf_refined.total_bounds)
print("RNP template bounds:", rnp_template_da.rio.bounds())
print("RNP boundary bounds:", rnp_gdf_refined.total_bounds)

# %% [markdown]
# ## STEP 4: Develop a fuzzy logic model
# 
# A fuzzy logic model is one that is built on expert knowledge rather than
# training data. You may wish to use the
# [`scikit-fuzzy`](https://pythonhosted.org/scikit-fuzzy/) library, which
# includes many utilities for building this sort of model. In particular,
# it contains a number of **membership functions** which can convert your
# data into values from 0 to 1 using information such as, for example, the
# maximum, minimum, and optimal values for soil pH.
# 
# To train a fuzzy logic habitat suitability model:</p>
# <pre><code>1. Find the optimal values for your species for each variable you are using (e.g. soil pH, slope, and current annual precipitation). 
# 2. For each **digital number** in each raster, assign a **continuous** value from 0 to 1 for how close that grid square/pixel is to the optimum range (1 = optimal, 0 = incompatible). 
# 3. Combine your layers by multiplying them together. This will give you a single suitability number for each grid square.
# 4. Optionally, you may apply a suitability threshold to make the most suitable areas pop on your map.</code></pre></div></div>
# 
# > **Tip**
# >
# > If you use mathematical operators on a raster in Python, it will
# > automatically perform the operation for every number in the raster.
# > This type of operation is known as a **vectorized** function. **DO NOT
# > DO THIS WITH A LOOP!**. A vectorized function that operates on the
# > whole array at once will be much easier and faster.

# %%
# Create a function that assigns suitability values for values within a suitable range
"""This function assigns values on a 0-1 scale. The closer a value is to one, the more suitable the habitat."""
def fuzzy_optimal_range(fuzzy_da, lower_optimum, upper_optimum, lower_limit, upper_limit):
    fuzzy_da = fuzzy_da.astype(np.float32)
    result = xr.where(
        fuzzy_da <= lower_limit,
        0.0,
        xr.where(
            fuzzy_da < lower_optimum,
            (fuzzy_da - lower_limit) / (lower_optimum - lower_limit),
            xr.where(
                fuzzy_da <= upper_optimum,
                1.0,
                xr.where(
                    fuzzy_da < upper_limit,
                    (upper_limit - fuzzy_da) / (upper_limit - upper_optimum),
                    0.0
                )
            )
        )
    )

    # Return the output of the function
    return result.clip(0.0, 1.0).astype(np.float32)

# %%
# Define fuzzy suitability functions for the following environmental variables:
"""Aspect is not included to avoid errors associated with including circular variables."""
# For soil pH
def fuzzy_ph(ph_da):
    ph_da = ph_da.astype(np.float32)  # Convert to float32 to save memory
    return fuzzy_optimal_range(
        ph_da,
        lower_optimum=5,
        upper_optimum=6.5,
        lower_limit=4,
        upper_limit=7
    )
# Source: (Roy, 1966)

# For soil bulk density
def fuzzy_bd(bd_da):
    bd_da = bd_da.astype(np.float32)  # Convert to float32 to save memory
    return fuzzy_optimal_range(
        bd_da,
        lower_optimum=1.3,
        upper_optimum=1.5,
        lower_limit=0.77,
        upper_limit=1.7
    )
# Source: (Lanman & Potter, 2025)

# For elevation
def fuzzy_elevation(elevation_da):
    elevation_da = elevation_da.astype(np.float32)  # Convert to float32 to save memory
    return fuzzy_optimal_range(
        elevation_da,
        lower_optimum=30,
        upper_optimum=760,
        lower_limit=0,
        upper_limit=1600
    )
# Source: (Olson Jr. et al, 1990)

# For slope
def fuzzy_slope(slope_da):
    slope_da = slope_da.astype(np.float32)  # Convert to float32 to save memory
    return fuzzy_optimal_range(
        slope_da,
        lower_optimum=0,
        upper_optimum=30,
        lower_limit=0,
        upper_limit=50
    )
# Source: (Lanman & Potter, 2025)

# For minimum temperature
def fuzzy_minimum_temperate(tasmin_da):
    tasmin_da = tasmin_da.astype(np.float32)  # Convert to float32 to save memory
    return fuzzy_optimal_range(
        tasmin_da,
        lower_optimum=9,
        upper_optimum=11,
        lower_limit=4,
        upper_limit=11
    )
# Source: (Olson Jr. et al., 1990)

# For maximum temperature
def fuzzy_maximum_temperature(tasmax_da):
    tasmax_da = tasmax_da.astype(np.float32)  # Convert to float32 to save memory
    return fuzzy_optimal_range(
        tasmax_da,
        lower_optimum=14.5,
        upper_optimum=19,
        lower_limit=14.5,
        upper_limit=25
    )
# Source: (Hellmers, 1966)

# For precipitation
def fuzzy_precipitation(pr_da):
    pr_da = pr_da.astype(np.float32) # Convert to float32 to save memory
    return fuzzy_optimal_range(
        pr_da,
        lower_optimum=1200,
        upper_optimum=2200,
        lower_limit=1200,
        upper_limit=4000
    )
# Source: (Sillett et al., 2025)

# %% [markdown]
# References
# 
# Hellmers, H. (1966). Growth Response of Redwood Seedlings to Thermoperiodism. *Forest Science*, 12(3), 276–283. https://doi.org/10.1093/forestscience/12.3.276
# 
# Lanman, R. B, & Potter, C. (2025). Machine learning model determines optimal coast redwood restoration sites in Santa Clara County, California. *bioRxiv*, 1-31. https://doi.org/10.1101/2025.03.18.644027
# 
# Olson, Jr., D. F., Roy, D. F., & Walters, G. A. (1990). Redwood Sequoia sempervirens (D. Don) Endl. Taxodiaceae—Redwood family. In R. M. Burns & B. H. Honkala (Eds.), *Silvics of North America: Volume 1. Conifers*. United States Forest Service. https://research.fs.usda.gov/silvics/redwood
# 
# Roy, D. F. (1966). *Silvical characteristics of Redwood (Sequoia sempervirens [D. Don] Endl.)*. U.S. Department of Agriculture, Forest Service, Pacific Southwest Forest and Range Experiment Station. https://www.fs.usda.gov/psw/publications/documents/psw_rp028/psw_rp028.pdf
# 
# Sillett, S. C., Chin, A. R., Carroll, A. L., Graham, M. E., & Antoine, M. E. (2025). Improved allometry and heartwood development of Sequoia sempervirens in secondary forests. *Forest Ecology and Management*, *593*, 122926. https://doi.org/10.1016/j.foreco.2025.122926

# %%
# Create a function that combines the fuzzy layers into a habitat suitability raster
def fuzzy_habitat_suitability(ph_da, bd_da, elevation_da, slope_da, tasmin_da, tasmax_da, pr_da):
    soil_ph_suitability = fuzzy_ph(ph_da)
    soil_bulk_density_suitability = fuzzy_bd(bd_da)
    elevation_suitability = fuzzy_elevation(elevation_da)
    slope_suitability = fuzzy_slope(slope_da)
    minimum_temperature_suitability = fuzzy_minimum_temperate(tasmin_da)
    maximum_temperature_suitability = fuzzy_maximum_temperature(tasmax_da)
    precipitation_suitability = fuzzy_precipitation(pr_da)

    # Multiply the suitability layers to create a habitat suitability data array
    habitat_suitability_da = (
        soil_ph_suitability *
        soil_bulk_density_suitability *
        elevation_suitability *
        slope_suitability *
        minimum_temperature_suitability *
        maximum_temperature_suitability *
        precipitation_suitability
    )

    # Write CRS back onto the output raster
    habitat_suitability_da = habitat_suitability_da.rio.write_crs(elevation_da.rio.crs)

    # Return the habitat suitability data array
    return habitat_suitability_da

# %%
# Create an empty list for the list of habitat suitability results for Redwood National Park
rnp_habitat_suitability_results = []

# Create a function to combine and append the list of habitat suitability results for Redwood National Park:
# Loop through each climate model
for model in models:

    # Loop through historical and future scenarios
    for scenario in ["Historical", "Future"]:
        # Extract the climate rasters
        # For minimum temperature
        tasmin_da = next(
            (
                item["average_climate_raster"]
                for item in aligned_rnp_climate
                if item["climate_model"] == model
                and item["scenario"] == scenario
                and item["climate_variable"] == "tasmin"
            ),
            None
        )
        # For maximum temperature
        tasmax_da = next(
            (
                item["average_climate_raster"]
                for item in aligned_rnp_climate
                if item["climate_model"] == model
                and item["scenario"] == scenario
                and item["climate_variable"] == "tasmax"
            ),
            None
        )
        # For precipitation
        # For maximum temperature
        pr_da = next(
            (
                item["average_climate_raster"]
                for item in aligned_rnp_climate
                if item["climate_model"] == model
                and item["scenario"] == scenario
                and item["climate_variable"] == "pr"
            ),
            None
        )
        # Extract the soil and topographic rasters
        ph_da = aligned_rnp_soil["ph"]
        bd_da = aligned_rnp_soil["bd"]
        elevation_da = aligned_rnp_topography["elevation"]
        slope_da = aligned_rnp_topography["slope"]

        # Skip over missing datasets
        if tasmin_da is None or tasmax_da is None:
            print(f"Missing data for {model} {scenario}")
            continue

        # Save habitat_suitability_da
        habitat_suitability_da = fuzzy_habitat_suitability(
            ph_da,
            bd_da,
            elevation_da,
            slope_da,
            tasmin_da,
            tasmax_da,
            pr_da
)

        # Append the results
        rnp_habitat_suitability_results.append({
            "climate_model": model,
            "scenario": scenario,
            "habitat_suitability": habitat_suitability_da
        })

# %%
# Create a function to combine and append the list of habitat suitability results for Humboldt Redwoods State Park:
hrsp_habitat_suitability_results = []

# Loop through each climate model
for model in models:

    # Loop through historical and future scenarios
    for scenario in ["Historical", "Future"]:
        # Extract the climate rasters
        # For minimum temperature
        tasmin_da = next(
            (
                item["average_climate_raster"]
                for item in aligned_hrsp_climate
                if item["climate_model"] == model
                and item["scenario"] == scenario
                and item["climate_variable"] == "tasmin"
            ),
            None
        )
        # For maximum temperature
        tasmax_da = next(
            (
                item["average_climate_raster"]
                for item in aligned_hrsp_climate
                if item["climate_model"] == model
                and item["scenario"] == scenario
                and item["climate_variable"] == "tasmax"
            ),
            None
        )
        # For precipitation
        pr_da = next(
            (
                item["average_climate_raster"]
                for item in aligned_hrsp_climate
                if item["climate_model"] == model
                and item["scenario"] == scenario
                and item["climate_variable"] == "pr"
            ),
            None
        )
        # Extract the soil and topographic rasters
        ph_da = aligned_hrsp_soil["ph"]
        bd_da = aligned_hrsp_soil["bd"]
        elevation_da = aligned_hrsp_topography["elevation"]
        slope_da = aligned_hrsp_topography["slope"]

        # Skip over missing datasets
        if tasmin_da is None or tasmax_da is None:
            print(f"Missing data for {model} {scenario}")
            continue

        # Save habitat_suitability_da
        habitat_suitability_da = fuzzy_habitat_suitability(
            ph_da,
            bd_da,
            elevation_da,
            slope_da,
            tasmin_da,
            tasmax_da,
            pr_da
)

        # Append the results
        hrsp_habitat_suitability_results.append({
            "climate_model": model,
            "scenario": scenario,
            "habitat_suitability": habitat_suitability_da
        })

# %% [markdown]
# ## STEP 5: Present your results
# Generate some plots that show your key findings of habitat suitability in your study sites across the different time periods and climate models. Don’t forget to interpret your plots!

# %%
# Create an insertable function that will plot the habitat suitability for Redwood National Park
def plot_habitat_suitability(habitat_suitability_results, rnp_gdf_refined, site_name):
    
    # Loop through each climate model
    for model in models:

        # Loop through historical and future scenarios
        for scenario in ["Historical", "Future"]:
            # Extract the habitat suitability raster for the current model and scenario
            habitat_suitability = next(
                (
                    item["habitat_suitability"]
                    for item in habitat_suitability_results
                    if item["climate_model"] == model
                    and item["scenario"] == scenario
                ),
                None
            )

            # Skip over missing habitat suitability rasters
            if habitat_suitability is None:
                print(f"Missing habitat suitability for {model} {scenario}")
                continue

            # Create a figure for the plot and specify its size
            fig, ax = plt.subplots(figsize=(8, 6))
            # Plot the habitat suitability raster
            habitat_suitability.plot(
                ax=ax,
                cmap="viridis", # Set the color scheme to maximize accessibility                
                vmin=0,
                vmax=1,
                cbar_kwargs={"label": "Habitat Suitability"} # Specify the metric being measured for the color bar label
            )
            # Reproject the site boundary to match the habitat suitability raster's CRS
            if habitat_suitability.rio.crs is not None:
                 rnp_gdf_reprojected = rnp_gdf_refined.to_crs(habitat_suitability.rio.crs)
            else:
                rnp_gdf_reprojected = rnp_gdf_refined
            # Overlay the site boundary
            rnp_gdf_refined.boundary.plot(
                ax=ax,
                color="black",
                linewidth=1
            )
            # Set the plot title using the time labels
            ax.set_title(
                f'Habitat Suitability in Redwood National Park Between {time_prop_labels[scenario]} as Based on ({model}) and RCP 8.5'
            )
            # Label the axes to reflect their spatial dimension
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            # Display the plot
            plt.show()

# %%
# Call the function to generate the habitat suitability plots for Redwood National Park
plot_habitat_suitability(
    habitat_suitability_results=rnp_habitat_suitability_results,
    rnp_gdf_refined=rnp_gdf_refined,
    site_name="Redwood National Park"
)

# %%
# Create an insertable function that will plot the habitat suitability for Humboldt Redwood State Park
def plot_habitat_suitability(habitat_suitability_results, hrsp_gdf_refined, site_name):
    
    # Loop through each climate model
    for model in models:

        # Loop through historical and future scenarios
        for scenario in ["Historical", "Future"]:
            # Extract the habitat suitability raster for the current model and scenario
            habitat_suitability = next(
                (
                    item["habitat_suitability"]
                    for item in habitat_suitability_results
                    if item["climate_model"] == model
                    and item["scenario"] == scenario
                ),
                None
            )

            # Skip over missing habitat suitability rasters
            if habitat_suitability is None:
                print(f"Missing habitat suitability for {model} {scenario}")
                continue

            # Create a figure for the plot and specify its size
            fig, ax = plt.subplots(figsize=(8, 6))
            # Plot the habitat suitability raster
            habitat_suitability.plot(
                ax=ax,
                cmap="viridis", # Set the color scheme to maximize accessibility                
                vmin=0,
                vmax=1,
                cbar_kwargs={"label": "Habitat Suitability"} # Specify the metric being measured for the color bar label
            )

            # Reproject the site boundary to match the habitat suitability raster's CRS
            if habitat_suitability.rio.crs is not None:
                 hrsp_gdf_reprojected = hrsp_gdf_refined.to_crs(habitat_suitability.rio.crs)
            else:
                hrsp_gdf_reprojected = hrsp_gdf_refined
            # Overlay the site boundary
            hrsp_gdf_refined.boundary.plot(
                ax=ax,
                color="black",
                linewidth=1
            )
            # Set the plot title using the time labels
            ax.set_title(
                f'Habitat Suitability in Humboldt Redwoods State Park Between {time_prop_labels[scenario]} as Based on ({model}) and RCP 8.5'
            )
            # Label the axes to reflect their spatial dimension
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            # Display the plot
            # Label axes
            plt.show()

# %%
# Call the function to generate the habitat suitability plots for Humboldt Redwoods State Park
plot_habitat_suitability(
    habitat_suitability_results=hrsp_habitat_suitability_results,
    hrsp_gdf_refined=hrsp_gdf_refined,
    site_name="Humboldt Redwoods State Park"
)

# %% [markdown]
# Interpret your plots here:

# %% [markdown]
# The climate scenario chosen for the habitat suitability analysis was RCP 8.5, which represents a high anthropogenic greenhouse emissions scenario that is projected to occur if actions are not taken to curb emissions (Riahi et al, 2011). The Multivariate Adaptive Constructed Analogs (MACA) data only accounts for how certain meteorological variables—namely, precipitation and temperature—are projected to change by different climate models (Joyce et al, 2018); though anthropogenic impacts like soil contamination can change soil pH and bulk density over time, soil properties are assumed to remain consistent throughout the timeframe (Dror et al, 2021). To avoid complications associated with including a circular variable like the slope aspect, it was excluded from the final habitat suitability analysis. 
# 
# California redwoods have evolved to suit the narrow belt of habitat they inhabit between central California and southern Oregon, and when considering shifting habitat suitability in the face of climate change, temperature and precipitation were the most important variables quantified. The average annual temperature for these redwood forests fluctuates between 10 and 16°C  (Olson. Jr. et al, 1990), but California redwood saplings display optimal growth around a base temperature of 19°C with a thermoperiod of 4°C (Hellmers, 1966). Estimating optimal precipitation range is difficult because of limited literature findings; precipitation typically falls between 640 and 3100 mm (Olson. Jr. et al, 1990), but these trees are noted to grow particularly well in a 1160-2200 mm precipitation range (Sillett et al., 2025). All climate models project precipitation that meets or exceeds this range, suggesting that, excluding extreme flood events, precipitation changes driven by climate change may not be a limiting factor for this site; MIROC-ESM-CHEM is the only model that showed a drop in precipitation, but modeled dryer conditions were not enough to stress California redwood populations. Temperature was the deciding factor for habitat suitability; curiously, rising temperatures made Redwood National Park more suitable for California redwoods, while the opposite held true for Humboldt Redwoods State Park. Historically, Redwood National Park is colder than Humboldt Redwoods State Park, and this hotter baseline would make it harder for the state park's California redwood population to tolerate rising temperatures.
# 
# The habitat suitability analysis does not show the full impact of climate change on California redwoods. MACA data does not quantify climate change's impact on fog, which is fundamental to how a California redwood gets water (Werner et al., 2020). The shapefile database was another limiting factor. Though care was taken to chose the southernmost site possible, California redwoods are found as far south as Los Padres National Forest. Choosing another database may have granted more insight into how climate change affects California redwoods, as southern California is anticipated to be more impacted by drought brought by climate change (Sillett et al., 2025).
# 
# References
# 
# Dror, I., Yaron, B., & Berkowitz, B. (2021). The Human Impact on All Soil-Forming Factors during the Anthropocene. *ACS Environmental Au*, *2*(1). https://doi.org/10.1021/acsenvironau.1c00010
# 
# Hellmers, H. (1966). Growth Response of Redwood Seedlings to Thermoperiodism. *Forest Science*, 12(3), 276–283. https://doi.org/10.1093/forestscience/12.3.276
# 
# Joyce, L. A., Abatzoglou, J. T., & Coulson, D. P.. (2018). *Climate data for RPA 2020 Assessment: MACAv2 (METDATA) historical modeled (1950-2005) and future (2006-2099) projections for the conterminous United States at the 1/24 degree grid scale*. Forest Service Research Data Archive. https://doi.org/10.2737/rds-2018-0014
# 
# Olson, Jr., D. F., Roy, D. F., & Walters, G. A. (1990). Redwood Sequoia sempervirens (D. Don) Endl. Taxodiaceae—Redwood family. In R. M. Burns & B. H. Honkala (Eds.), *Silvics of North America: Volume 1. Conifers*. United States Forest Service. https://research.fs.usda.gov/silvics/redwood
# 
# Riahi, K., Rao, S., Krey, V., Cho, C., Chirkov, V., Fischer, G., Kindermann, G., Nakicenovic, N., & Rafaj, P. (2011). RCP 8.5—A scenario of comparatively high greenhouse gas emissions. *Climatic Change*, *109*(1-2), 33–57. https://doi.org/10.1007/s10584-011-0149-y
# 
# Sillett, S. C., Chin, A. R., Carroll, A. L., Graham, M. E., & Antoine, M. E. (2025). Improved allometry and heartwood development of Sequoia sempervirens in secondary forests. *Forest Ecology and Management*, *593*, 122926. https://doi.org/10.1016/j.foreco.2025.122926
# 
# Werner, Z., Berger, A., Winter, A., Choi, C. T. H., Evangelista, P., Jarnevich, C., Vorster, T., Woodward, B., & Young N. (2020). *California & Oregon Ecological Forecasting: Detecting and Forecasting Fog Occurrence, Frequency, and Change to Support Coast Redwood (Sequoia sempervirens) Habitat Assessments*. https://ntrs.nasa.gov/api/citations/20205011382/downloads/2020Fall_CO_California%26OregonEco_ProjectSummary_FD-final.docx.pdf


