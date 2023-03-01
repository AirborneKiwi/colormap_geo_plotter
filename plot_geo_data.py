"""
This script can plot numeric data from a simple csv table into a map consisting of several layers of shape files.
Those shape files can be found in the geo_data folder.

It used the headers of the csv file and will look for a county in the
shape file 'geo_data/vg5000_01-01.gk3.shape.ebenen/vg5000_ebenen_0101/VG5000_KRS.shp' with a mathing name.
Afterwards it uses the data value to set the fill color of the county corresponding to a chosen colormap.

Each row in the data will be saved to a separate modified svg-file.

Arguments:
    -h
        Display this help message^^

    -f <datafile>, --filename <datafile>
        Default "ExampleData.csv". The filename of the data file that shall be plotted.
            If you want to specify the location using region names, use the following format:
                'Region 1', 'Region 2', 'Region 3'
                    0.2   ,    0.1    ,    0.3
                    0.3   ,    0.2    ,    0.5
            If you want labeled circles of 'value'-dependant size and color at specific coordinates:
                 'lat', 'lon' , 'value', 'label'
                49.294, 10.663,    1   ,  'P0'
                49.258, 10.664,    2   ,  'P1'
                50.380, 11.957,   3.2  ,  'P2'
            And if you want non labeled circles of 'value'-dependant color at specific coordinates:
                 'lat', 'lon' , 'value', 'label'
                49.294, 10.663,  'eins',  'P0'
                49.258, 10.664,  'zwei',  'P1'
                50.380, 11.957,  'drei',  'P2'

    -a <yourtitle>, --title_axis <yourtitle>
        Default "Please add a title to the colorbar!". The title for the colorbar.

    -T <yourtitle>, --figure_title <youttitle>
        Optional. The title of the whole figure.

    -t <yourtitle>, --title <yourtitle>
        Optional. The title of the each subfigure. Can be a semicolon-separated list, like "title 0;title 1"

    --cmap <yourcmap>
        Default "RdYlGn". The matplotlib colormap of your choice.
        You can reverse a colormap, by adding a '_r' suffix to a cmap.

    --text_alpha <flaot>
        Default 0.5. The alpha value of text elements. 0 is invisible.

    --save_to <yourPath>
        Default 'output/{datafile}'. The folder and filename of the generated image files.

    --format <imageFormat>
        Default 'png'. The image file format of your choice. Recommended choices are 'png', 'svg' or 'pdf'.

    --crop' <yourChoice>
        Default "9.9136935; 12.7207591; 48.7730094; 50.6524235".
            If you pass True or 'data', the map will be cropped to fit the data in datafile.
            Alternatively you can pass a list of coordinates describing the geo-referenced bounding box in
            semicolon-separated decimal longitude and latitude values.
            The bounding box for the European Metropolitan Area Nuremberg (EMN) is "9.9136935, 12.7207591, 48.7730094, 50.6524235"
            If you pass False, the all data and shapefiles will be plotted (Not recommended!)

    --zoom <float>
        Default 1. A factor that scales all linewidth, textsizes and cicles. Can be used to finetune those to different map sizes.

    --data_viz <yourChoice>
        Default 'area'.
            If you pass 'area', the values from the datafile will be used to color the corresponding regions.
            If you pass 'circle', instead of coloring the regions a circles will be drawn at the center of the
            corresponding regions. Their size and color depends on the data.
            Automatically set to 'circle', if the data has a 'lat' and a 'lon' column.

    --label_counties
        Default True. Defines whether counties will be labeled or not.

    --add_roads
        Default False. Defines whether roads will be plotted or not.

    --use_cx
        Default False. Defines whether labels will be loaded from the provider Stamen using the contextly library.

    --title_size <yourChoice>
        Default "medium". Defines the font size of the title of the sub-figures.
            Valid font size is numeric values or "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large", "larger", "smaller", "None".

    --cbar_label_size <yourChoice>
        Default "medium". Defines the font size of the title of the colorbar.
            Valid font size is numeric values or "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large", "larger", "smaller", "None".

    --cbar_tick_size <yourChoice>
        Default "medium". Defines the font size of the ticks of the colorbar.
            Valid font size is numeric values or "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large", "larger", "smaller", "None".

    --fig_title_size <yourChoice>
        Default "medium". Defines the font size of the figure title.
            Valid font size is numeric values or "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large", "larger", "smaller", "None".

    --markersize <float>
        The size of markers, when using data_viz="circle" and there are no values to scale the marker.

    --markersize_min <float>
        The minimum size of markers, when using data_viz="circle" and there are values to scale the marker.

    --markersize_max <float>
        The maximum size of markers, when using data_viz="circle" and there are values to scale the marker.

    --no_value_labels
        Default False. Add no value labels for each datapoint.

    --label_size <float>
        The font size of the value labels.
"""


# !/usr/bin/python
import sys
import getopt
import os

import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import transform
import pyproj

def save_figure(fig:plt.Figure, filename:str, format:str = 'png') -> None:
    # Create the output folder, if it does not exist
    path = filename[:filename.rfind('/')]
    if path:
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    print(f'Saving to {filename}.{format}')
    fig.savefig(f'{filename}.{format}', format=format, bbox_inches='tight', dpi=600)

def create_coordinate(longitude:float, latitude:float) -> Point:
    return gpd.points_from_xy([longitude], [latitude], crs='EPSG:4326')[0]

def create_bounding_box_from_lon_lat(sw_lon:float,  ne_lon:float, sw_lat:float, ne_lat:float) -> Polygon:
    return create_bounding_box(create_coordinate(sw_lon, sw_lat), create_coordinate(ne_lon, ne_lat))

def create_bounding_box(sw_point:Point, ne_point:Point) -> Polygon:
    return Polygon([(sw_point.x, sw_point.y), (ne_point.x, sw_point.y), (ne_point.x, ne_point.y), (sw_point.x, ne_point.y), (sw_point.x, sw_point.y)])

def plot_geo_data(data: pd.DataFrame, title_axis: str, data_viz:str = 'area', title: str = '', cmap: str = 'RdYlGn',
                  text_alpha: float = 0.5, label_counties: bool = True, crop: object = 'data', add_roads: bool = False,
                  use_cx: bool = False, save_to: str = '', format: str = 'svg', vmin=None, vmax=None, fig=None, ax=None,
                  zoom=1, **kwargs) -> (plt.Figure, plt.Axes):
    if use_cx:
        import contextily as cx

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(15, 15))

    ax.set_axis_off()

    data_copy = data
    coord_specified = 'lat' in data_copy and 'lon' in data_copy

    if coord_specified and 'value' not in data_copy:
        data_copy['value'] = 1

    if all(data['value'].isna()):
        print('No data to plot found')
        return fig, ax

    ax.set_title(title, fontdict=dict(fontweight='bold', size=(kwargs['title_size'] if 'title_size' in kwargs else 'medium')), y=1.02)

    '''Load the shape files'''
    if add_roads:
        geo_data_roads = gpd.read_file('geo_data/dlm1000.gk3.shape.ebenen/dlm1000_ebenen/ver01_l.shp').to_crs(epsg=3857)
        geo_data_roads = geo_data_roads[(geo_data_roads['BEZ'].str.contains('A').fillna(False)) | (
            geo_data_roads['BEZ'].str.contains('E').fillna(False))]

    geo_data_sta = gpd.read_file('geo_data/vg5000_01-01.gk3.shape.ebenen/vg5000_ebenen_0101/VG5000_STA.shp').to_crs(
        epsg=3857)
    geo_data_lan = gpd.read_file('geo_data/vg5000_01-01.gk3.shape.ebenen/vg5000_ebenen_0101/VG5000_LAN.shp').to_crs(
        epsg=3857)
    geo_data_krs = gpd.read_file('geo_data/vg5000_01-01.gk3.shape.ebenen/vg5000_ebenen_0101/VG5000_KRS.shp').to_crs(
        epsg=3857)
    geo_data_krs['centroids'] = geo_data_krs.centroid

    '''Create the plottable DataFrame with the data'''

    data_df = None
    if not coord_specified:
        '''Try to find matching counties in geo_data_krs for each column header'''
        marking = ''
        def is_county_marked(name):
            return re.match(r'.*\bLK\b.*', name) or re.match(r'.*\bLandkreis\b.*', name) or \
                   re.match(r'.*\bKreis\b.*', name) or re.match(r'.*\bK\b.*', name)

        def is_city_marked(name):
            return re.match(r'.*\bStadt\b.*', name) or re.match(r'.*\bKreisstadt\b.*', name)

        if 'loc' not in data_copy:
            data_copy['loc'] = data_copy.index.values

        locations = data_copy['loc'].values

        for name in locations:
            if is_county_marked(name):
                marking = 'County'
                break

            if is_city_marked(name):
                marking = 'City'
                break

        if not marking:
            raise Exception('Please either use names with "LK " or cities with "Stadt " to distinguish them.')

        def get_type(name, marking):
            if marking == 'County':
                return 'County' if is_county_marked(name) else 'City'

            return 'City' if is_city_marked(name) else 'County'

        # data_copy['area_type'] = data_copy['name'].apply(lambda s: get_type(s, marking))
        data_copy['area_type'] = [get_type(name, marking) for name in locations]  # data_copy['name'].apply(lambda s: get_type(s, marking))

        def match_to_krs(s):
            def get_name_matches(s_krs, name, area_type):
                name_parts = name.split(' ')
                score = 0

                # if area_type == 'County' and s_krs['BEZ']!='Landkreis' and s_krs['BEZ']!='Kreis':
                #    score -= 0.25

                # if area_type == 'City' and (s_krs['BEZ']=='Landkreis' or s_krs['BEZ']=='Kreis'):
                #    score -= 0.25

                d_score = 2

                for name_part in name_parts:
                    if name_part in ['Landkreis', 'LK', 'Kreis', '(Landkreis)', '(LK)', '(Kreis)']:
                        continue

                    d_score *= 0.5
                    if re.match(fr'.*\b{name_part}\b.*', s_krs['GEN']):
                        score += d_score
                        continue

                    if re.match(fr'.*\b{name_part}.*', s_krs['GEN']):
                        score += d_score
                        continue

                return score

            area_type = s['area_type']
            name = s['loc']

            geo_data_krs['score'] = geo_data_krs.apply(lambda s_krs: get_name_matches(s_krs, name, area_type), axis=1)

            match = geo_data_krs[(geo_data_krs['score'] > 0) & (geo_data_krs['score'] == geo_data_krs['score'].max())]

            if len(match) == 0:
                return np.nan

            if len(match) > 1:  # multiple matches found. Try to distinguish using the area type
                if area_type == 'County':
                    match = match[(match['BEZ'] == 'Landkreis') | (match['BEZ'] == 'Kreis')]

                if area_type == 'City':
                    match = match[(match['BEZ'] != 'Landkreis') & (match['BEZ'] != 'Kreis')]

                if len(match) > 1:
                    raise Exception(
                        f'No clear match for {area_type} "{name}" found. Conflicting entries are {list(zip(match["GEN"].values, match["BEZ"].values))}')
            #print(f'{name} matched to {match.at[match.index[0], "BEZ"]} {match.at[match.index[0], "GEN"]}')
            return match.index[0]

        data_copy.index = data_copy.apply(match_to_krs, axis=1)

        data_df = geo_data_krs.join(data_copy, rsuffix='_data')

        data_df = data_df.drop(['area_type', 'score'], axis=1)
    else:
        '''Extract latitude and longitude and create geopandas frame'''
        data_viz = 'circle'
        data_df = gpd.GeoDataFrame(data_copy, geometry=gpd.points_from_xy(data_copy['lon'], data_copy['lat'],
                                                                          crs='EPSG:4326')).to_crs(epsg=3857)


    data_df['has_data'] = data_df['value'].notna()
    data_df['centroids'] = data_df.centroid

    '''Plot the data'''
    if vmin is None:
        if 'value' in data_df:
            vmin = data_df[data_df['has_data']]['value'].min()
        else:
            vmin = data_df[data_df['has_data']].min().min()

    if vmax is None:
        if 'value' in data_df:
            vmax = data_df[data_df['has_data']]['value'].max()
        else:
            vmax = data_df[data_df['has_data']].max().max()

    add_colorbar = own_fig

    if data_viz == 'area':
        ax = data_df[data_df['has_data']].plot(ax=ax, column='value', linewidth=0, zorder=3, legend=False,
                                                         vmin=vmin, vmax=vmax, cmap=cmap,
                                               scheme=None, # quantiles, natural_breaks, equal_interval
                                               missing_kwds={
                                                   "color": "lightgrey",
                                                   "edgecolor": "red",
                                                   "hatch": "///",
                                                   "label": "Missing values",
                                               }
                                               )
    elif data_viz == 'circle':
        ax = data_df[data_df['has_data']].plot(ax=ax, facecolor="none", edgecolor="none", zorder=3)
        point_data = data_df[data_df['has_data']].drop(columns=['geometry']).rename(columns={'centroids': 'geometry'})

        all_numeric = all(point_data['value'].apply(lambda v: isinstance(v, int) or isinstance(v, float)))

        if all_numeric and len(point_data['value'].unique()) > 1:
            point_data['scaled_value'] = (point_data['value'] - point_data['value'].min()) / (point_data['value'].max() - point_data['value'].min())

            # make sure smaller circles are plotted on top
            point_data = point_data.sort_values(by='scaled_value', ascending=0)

            markersize_min = kwargs['markersize_min'] if 'markersize_min' in kwargs else 100
            markersize_max = kwargs['markersize_max'] if 'markersize_max' in kwargs else 5000
            point_data['markersize'] = markersize_min+(markersize_max-markersize_min)*point_data['scaled_value']
        else:
            point_data['markersize'] = kwargs['markersize'] if 'markersize' in kwargs else 100
            add_colorbar = False

        ax = point_data.plot(ax=ax, column='value', edgecolor="black", linewidth=zoom*1,
                             markersize=zoom*point_data['markersize'],
                             zorder=18, cmap=cmap, legend = not all_numeric)

        if not all_numeric:
            legend = plt.gca().legend_
            if legend is not None:
                legend.set_title(title_axis)

        if 'label' in point_data:
            for x, y, label in zip(point_data['geometry'].x, point_data['geometry'].y, point_data['label']):
                ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", size=zoom * (kwargs['label_size'] if 'label_size'  in kwargs else 12), alpha=text_alpha, zorder=20)
        pass

    add_value_labels = True if 'no_value_labels' not in kwargs else False
    if add_value_labels:
        data_df_with_data = data_df[data_df['has_data']]
        for centroid, value in zip(data_df_with_data['geometry'].centroid, data_df_with_data['value']):
            color = 'black'
            zero_middled = False
            if zero_middled:
                if value > vmin + 0.8*vmax or  value < vmin + 0.8*vmax:
                    color = 'white'
            else:
                if value > vmin + 0.8 * (vmax - vmin):
                    color = 'white'

            ax.annotate(f'{int(value)}',
                        xy=(centroid.x, centroid.y), xytext=(0, 0),
                        textcoords="offset points", va='center', ha='center',
                        color=color,
                        size=zoom * (kwargs['label_size'] if 'label_size' in kwargs else 12), alpha=1,
                        zorder=25)

    if own_fig and add_colorbar:
        cax = fig.add_axes([1, 0.1, 0.03, 0.8])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbr = fig.colorbar(sm, cax=cax)
        cbr.ax.tick_params(labelsize=(kwargs['cbar_tick_size'] if 'cbar_tick_size' in kwargs else 'medium'))
        cbr.set_label(label=title_axis, size=(kwargs['cbar_label_size'] if 'cbar_label_size' in kwargs else 'medium'))


    xlim = None
    ylim = None
    if crop == 'data' or crop==True:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    elif isinstance(crop, Polygon) or isinstance(crop, list) or isinstance(crop, set):
        if isinstance(crop, list) or isinstance(crop, set):
            crop = create_bounding_box_from_lon_lat(*crop)

        print(f'Cropping to bounding box {crop}')
        '''A bounding box has been passed'''
        wgs84 = pyproj.CRS('EPSG:4326')
        pseudo_mercator = pyproj.CRS('EPSG:3857')

        project = pyproj.Transformer.from_crs(wgs84, pseudo_mercator, always_xy=True).transform
        crop = transform(project, crop)

        minx, miny, maxx, maxy = crop.bounds
        xlim = (minx, maxx)
        ylim = (miny, maxy)

    '''Border plotting'''
    # ax = geo_data_vwg.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.1, zorder=4)
    ax = geo_data_krs.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=zoom*0.2, zorder=5)
    ax = geo_data_lan.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=zoom*1, zorder=6)
    ax = geo_data_sta.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=zoom*3, zorder=7)

    '''Road plotting'''
    if add_roads:
        ax = geo_data_roads[geo_data_roads['BEZ'].str.contains('E')].plot(ax=ax, edgecolor="green", linewidth=2.5,
                                                                          zorder=13)
        ax = geo_data_roads[geo_data_roads['BEZ'].str.contains('E')].plot(ax=ax, edgecolor="black", linewidth=3,
                                                                          zorder=12)

        ax = geo_data_roads[geo_data_roads['BEZ'].str.contains('A')].plot(ax=ax, edgecolor="orange", linewidth=3,
                                                                          zorder=15)
        ax = geo_data_roads[geo_data_roads['BEZ'].str.contains('A')].plot(ax=ax, edgecolor="black", linewidth=3.5,
                                                                          zorder=14)

    if xlim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    '''Add labels'''
    if use_cx:
        cx.add_basemap(ax, source=cx.providers.Stamen.TonerLabels, zorder=10, zoom=8, alpha=text_alpha)
    else:
        if label_counties:
            geo_data_counties = geo_data_krs[(geo_data_krs['BEZ'] != 'Kreisfreie Stadt')]
            geo_data_counties = geo_data_counties.drop('geometry', axis=1).rename(columns={'centroids': 'geometry'})

            for x, y, label in zip(geo_data_counties['geometry'].x, geo_data_counties['geometry'].y,
                                   geo_data_counties['GEN']):
                ax.annotate(f'LK {label}', xy=(x, y), xytext=(0, 0), textcoords="offset points", va='center',
                            ha='center', size=zoom*6, alpha=text_alpha, zorder=20)

        geo_data_cities = geo_data_krs[(geo_data_krs['BEZ'] == 'Kreisfreie Stadt')]
        geo_data_cities = geo_data_cities.drop('geometry', axis=1).rename(columns={'centroids': 'geometry'})

        ax = geo_data_cities.plot(ax=ax, color='black', markersize=zoom*5, zorder=12)
        for x, y, label in zip(geo_data_cities['geometry'].x, geo_data_cities['geometry'].y, geo_data_cities['GEN']):
            ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points", size=zoom*10, alpha=text_alpha,
                        fontweight='bold', zorder=21)

    if save_to is not None and save_to != '':
        # Create the output folder, if it does not exist
        save_to.replace('\\', '/')
        save_figure(fig, save_to, format)
    return fig, ax



def process_data(df: pd.DataFrame, n_rows=1, n_cols=1, zoom=1, zero_middled=False, figure_title:str='', **kwargs) -> None:
    fig=None
    axes=[None]*len(df)
    individual_files = True

    if 'lat' in df and 'lon' in df:
        fig, axes = plot_geo_data(df, zoom=zoom, **kwargs)
    else:

        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
        else:
            vmin = df.min().min()

        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
        else:
            vmax = df.max().max()

        if zero_middled:
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax

        if n_rows != 1 or n_cols != 1:
            if len(df) != n_rows*n_cols:
                raise Exception(f'The dimensions of the subplots of {n_rows}x{n_cols} have been passed, but there are {len(df)} rows in the data!')
            individual_files = False

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

        for index, row in df.iterrows():
            if not individual_files:
                ax = axes.flatten()[index]
            else:
                ax = None

            tmp = pd.DataFrame(row)
            tmp = tmp.rename(columns={index: 'value'})
            args = kwargs.copy()
            args['vmin'] = vmin
            args['vmax'] = vmax

            if isinstance(args['title'], str):
                args['title'] = f'{kwargs["title"]} - {index}'
            else:
                try:
                    args['title'] = args['title'][index]
                except IndexError:
                    raise IndexError('The title can be a str or an iterable of the same size as there are number or rows.')

            if individual_files and args['save_to'] is not None:
                if isinstance(args['save_to'], str):
                    args['save_to'] = f'{kwargs["save_to"]}_{index}'
                else:
                    try:
                        args['save_to'] = args['save_to'][index]
                    except IndexError:
                        raise IndexError(
                            'The filename for the output (save_to) can be a str or an iterable of the same size as there are number or rows.')
            else:
                args['save_to'] = None
            fig, ax = plot_geo_data(tmp, fig=fig, ax=ax, zoom=zoom, **args)

        if n_rows != 1 or n_cols != 1:
            axes_pos = np.around(np.array([ax.get_position().get_points().flatten() for ax in fig.axes if ax.has_data()]).transpose(), decimals=10)

            # Get the height and width of axes with data
            ax_width = axes_pos[2][0] - axes_pos[0][0]
            ax_height = axes_pos[3][0] - axes_pos[1][0]

            y0_rows = np.unique(axes_pos[1])
            x0_cols = np.unique(axes_pos[0])

            # rescale and reposition all empty axes
            for ax in fig.axes:
                if not ax.has_data():
                    ax_pos = ax.get_position()
                    ax_pos.x0 = x0_cols[np.absolute(x0_cols - ax_pos.x0).argmin()]
                    ax_pos.y0 = y0_rows[np.absolute(y0_rows - ax_pos.y0).argmin()]
                    ax_pos.x1 = ax_pos.x0 + ax_width
                    ax_pos.y1 = ax_pos.y0 + ax_height
                    ax.set_position(ax_pos)


            # Compute the gap between the rightmost axes and the colorbar
            x_gap_width_min = 0.025
            x1_max = axes_pos[2].max()
            x0_min = axes_pos[0].min()

            x_gap_width = max(x_gap_width_min, (x1_max - x0_min)/len(x0_cols) - ax_width)


            # Reposition all subplots vertically
            max_y1 = 0.95
            if figure_title:
                max_y1 -= 0.05  # leave some space for the title

            y_gap_width = 0.025
            if 'title_axis' in kwargs:
                y_gap_width += 0.01  # leave some extra space for the axis title

            for ax in fig.axes:
                ax_col = np.absolute(x0_cols - ax.get_position().x0).argmin()
                ax_row = np.absolute(y0_rows - ax.get_position().y0).argmin()
                ax_pos = ax.get_position()
                ax_pos.x0 = x0_min + ax_col*(ax_width + x_gap_width)
                ax_pos.x1 = ax_pos.x0 + ax_width

                ax_pos.y1 = max_y1 - (n_rows-ax_row-1)*(ax_height + y_gap_width)
                ax_pos.y0 = ax_pos.y1 - ax_height
                ax.set_position(ax_pos)


            # add the colorbar
            x0_colorbar = x1_max
            if n_cols == 1:
                x0_colorbar += x_gap_width
            y0_colorbar = max_y1 - (n_rows*(ax_height + y_gap_width) - y_gap_width)
            y1_colorbar = (n_rows*(ax_height + y_gap_width) - y_gap_width)

            cax = fig.add_axes([x0_colorbar, y0_colorbar, 0.03, y1_colorbar])

            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'RdYlGn'

            sm = plt.cm.ScalarMappable(cmap=kwargs['cmap'], norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            cbr = fig.colorbar(sm, cax=cax)
            cbr.ax.tick_params(labelsize=(kwargs['cbar_tick_size'] if 'cbar_tick_size' in kwargs else 'medium'))
            cbr.set_label(label=kwargs['title_axis'], size=(kwargs['cbar_label_size'] if 'cbar_label_size' in kwargs else 'medium'))

    # add the figure title
    if figure_title:
        fig.suptitle(figure_title, y=0.95, fontweight='bold', fontsize=(kwargs['fig_title_size'] if 'fig_title_size' in kwargs else 16))

    # save to a file
    if not individual_files and kwargs['save_to'] is not None:
        save_figure(fig, kwargs['save_to'], kwargs['format'])

    return fig, axes

def process_file(filename: str, **kwargs) -> None:
    df = pd.read_csv(filename)
    print('The following data has been read:')
    print(df)
    process_data(df, **kwargs)

def example_color_regions():
    """
    This method plots the data from ExampleData.csv as colored regions onto a map.
    """
    _argv = [
        '-f', 'ExampleData.csv',
        '-a', 'Please add a title to the colorbar!',
        '-t' ,'Random example data',
        '--cmap', 'RdYlGn',
        '--text_alpha', '0.5',
        '--label_counties',
        '--crop', 'True',
        '--save_to', 'output/ExampleData',
        '--format', 'png'
        ]
    main(_argv)

def example_colored_circles_in_regions_single_figure():
    """
        This method plots the data from ExampleData.csv as colored circles onto a map using a single figure with two rows.
    """
    _argv = [
        '-f', 'ExampleData.csv',
        '-a', 'Please add a title to the colorbar!',
        '-T', 'Random example data, but as circles',
        '-t', 'Axis title 0;Axis title 1',
        '--label_counties',
        '--save_to', 'output/ExampleData_circles',
        '--data_viz', 'circle',
        '--n_rows', '2'
    ]

    main(_argv)

def example_color_regions_single_figure():
    """
        This method plots the data from ExampleData.csv as colored regions onto a map using only a single figure with two columns.
    """
    _argv = [
        '-f', 'ExampleData.csv',
        '-a', 'Please add a title to the colorbar!',
        '-T', 'Random example data',
        '-t', 'Axis title 0;Axis title 1',
        '--label_counties',
        '--n_cols', '2'
    ]
    main(_argv)

def example_points():
    """
        This method plots the data from ExampleData_points_1.csv as points onto a map.
    """
    _argv = [
        '-f', 'ExampleData_points_1.csv',
        '-T', 'Random points',
        '--save_to', 'output/Random_points_1',
        '--add_roads'
    ]
    main(_argv)

def example_points_with_values():
    """
        This method plots the data from ExampleData_points_2.csv as colored points onto a map.
    """
    _argv = [
        '-f', 'ExampleData_points_2.csv',
        '-T', 'Random points with values',
        '-a', 'Value',
        '--save_to', 'output/Random_points_2',
        '--add_roads'
    ]
    main(_argv)

def example_points_with_category_values():
    """
        This method plots the data from ExampleData_points_3.csv as colored points onto a map.
    """
    _argv = [
        '-f', 'ExampleData_points_3.csv',
        '-T', 'Random points with category values',
        '-a', 'Categories',
        '--save_to', 'output/Random_points_3',
        '--add_roads'
    ]
    main(_argv)

def example_points_with_category_values_and_labels():
    """
        This method plots the data from ExampleData_points_4.csv as colored points onto a map.
    """
    _argv = [
        '-f', 'ExampleData_points_4.csv',
        '-T', 'Random points with category values and labels',
        '-a', 'Categories',
        '--save_to', 'output/Random_points_4',
        '--add_roads'
    ]
    main(_argv)

def __extract_argv__(argv):
    _args = dict(
        filename='ExampleData.csv',
        title_axis='Please add a title to the colorbar!',
        figure_title='',
        title='',
        cmap='RdYlGn',
        text_alpha=0.5,
        label_counties=False,
        crop=[9.9136935, 12.7207591, 48.7730094, 50.6524235], # bbox of European Metropolitan Area Nuremberg
        add_roads=False,
        use_cx=False,
        save_to=None,
        format='png',
        data_viz='area'
    )


    def raise_error(opt, arg):
        raise ValueError(f'Warning! The value {arg} for the {opt} argument is not supported. See -h for possible values.')

    try:
        opts, args = getopt.getopt(argv, '"hf:a:t:T:o:',
                                   [
                                       'filename=',
                                       'title_axis=',
                                       'cbar_label_size=',
                                       'cbar_tick_size=',
                                       'title=',
                                       'title_size=',
                                       'figure_title=',
                                       'fig_title_size=',
                                       'cmap=',
                                       'text_alpha=',
                                       'save_to=',
                                       'format=',
                                       'crop=',
                                       'zoom=',
                                       'data_viz=',
                                       'n_rows=',
                                       'n_cols=',
                                       'label_counties',
                                       'add_roads',
                                       'use_cx',
                                       'markersize=',
                                       'markersize_min=',
                                       'markersize_max=',
                                       'no_value_labels',
                                       'label_size='
                                   ])
    except getopt.GetoptError as e:
        print(e)
        print('Refer to the help:')
        print(__doc__)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(__doc__)
            sys.exit()
        elif opt in ("-f", "--filename"):
            _args['filename'] = arg
        elif opt in ("-T", "--figure_title"):
            _args['figure_title'] = arg
        elif opt in ("--fig_title_size"):
            if arg.isnumeric():
                _args['fig_title_size'] = float(arg)
            else:
                _args['fig_title_size'] = arg
        elif opt in ("-a", "--title_axis"):
            _args['title_axis'] = arg
        elif opt in ("--cbar_label_size"):
            if arg.isnumeric():
                _args['cbar_label_size'] = float(arg)
            else:
                _args['cbar_label_size'] = arg
        elif opt in ("--cbar_tick_size"):
            if arg.isnumeric():
                _args['cbar_tick_size'] = float(arg)
            else:
                _args['cbar_tick_size'] = arg
        elif opt in ("-t", "--title"):
            if ';' in arg:
                _args['title'] = arg.split(';')
            else:
                _args['title'] = arg
        elif opt in ("--title_size"):
            if arg.isnumeric():
                _args['title_size'] = float(arg)
            else:
                _args['title_size'] = arg
        elif opt in ("--cmap"):
            _args['cmap'] = arg
        elif opt in ("--format"):
            _args['format'] = arg
        elif opt in ("-o", "--save_to"):
            _args['save_to'] = arg
        elif opt in ("--text_alpha"):
            _args['text_alpha'] = float(arg)
        elif opt in ("--zoom"):
            _args['zoom'] = float(arg)
        elif opt in ("--n_rows"):
            _args['n_rows'] = int(arg)
        elif opt in ("--n_cols"):
            _args['n_cols'] = int(arg)
        elif opt in ("--label_counties"):
            _args['label_counties'] = True
        elif opt in ('markersize='):
            _args['markersize'] = float(arg)
        elif opt in ('markersize_min='):
            _args['markersize_min'] = float(arg)
        elif opt in ('markersize_max='):
            _args['markersize_max'] = float(arg)
        elif opt in ('no_value_labels'):
            _args['no_value_labels'] = False
        elif opt in ('label_size='):
            _args['label_size'] = float(arg)
        elif opt in ("--add_roads"):
            _args['add_roads'] = True
        elif opt in ("--use_cx"):
            _args['use_cx'] = True
        elif opt in ("--crop"):
            if ';' in arg:
                # expecting a string with decimal lon/lat values "[lon_min; lon_max; lat_min; lat_max]"
                bbox = arg.replace(' ', '').split(';')
                for i in range(len(bbox)):
                    bbox[i] = float(bbox[i])
                _args['crop'] = bbox
            elif arg in ['True', 'true', '1', 'data']:
                _args['crop'] = 'data'
            elif arg in ['False', 'false', '0']:
                _args['crop'] = False
            else:
                raise_error(opt, arg)
        elif opt in ("--data_viz"):
            if arg in ['area', 'circle']:
                _args['data_viz'] = arg
            else:
                raise_error(opt, arg)


    if _args['save_to'] is None:
        _args['save_to'] = f'output/{_args["filename"][:_args["filename"].rfind(".")]}'

    return _args

def main(argv):
    _args = __extract_argv__(argv)

    print(f'The data file is {_args["filename"]}')
    if _args["figure_title"] != '':
        print(f'The figure title is {_args["figure_title"]}')
    print(f'The titles is {_args["title"]}')
    print(f'The axis title is {_args["title_axis"]}')
    print(f'The colormap is {_args["cmap"]}')

    print(f'\nPlotting data from the csv file and saving each line to an individual {_args["format"]}-file')
    process_file(**_args)
    print('All saved.')


if __name__ == '__main__':
    main(sys.argv[1:])
