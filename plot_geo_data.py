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

    -d <datafile>, --datafile <datafile>
        Default "ExampleData.csv". The filename of the data file that shall be plotted.

    -a <yourtitle>, --title_axis <yourtitle>
        Default "Please add a title to the colorbar!". The title for the colorbar.

    -t <yourtitle>, --title <yourtitle>
        Optional. The title of the whole figure.

    --cmap <yourcmap>
        Default "RdYlGn". The matplotlib colormap of your choice.
        You can reverse a colormap, by adding a '_r' suffix to a cmap.
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

# Some libraries to render the svg to PDF and afterwards to PNG
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from pdf2image import convert_from_path

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

def plot_geo_data(data: pd.DataFrame, title_axis: str, title: str = '', cmap: str = 'RdYlGn',
                  text_alpha: float = 0.5, label_counties: bool = True, crop: bool = True, add_roads: bool = False,
                  use_cx: bool = False, save_to: str = '', format: str = 'svg', vmin=None, vmax=None, fig=None, ax=None, zoom=1) -> (plt.Figure, plt.Axes):
    if use_cx:
        import contextily as cx

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(15, 15))

    ax.set_axis_off()

    if all(data['value'].isna()):
        print('No data to plot found')
        return fig, ax

    data_copy = data

    marking = ''

    def is_county_marked(name):
        return re.match(r'.*\bLK\b.*', name) or re.match(r'.*\bLandkreis\b.*', name) or \
               re.match(r'.*\bKreis\b.*', name) or re.match(r'.*\bK\b.*', name)

    def is_city_marked(name):
        return re.match(r'.*\bStadt\b.*', name) or re.match(r'.*\bKreisstadt\b.*', name)

    for name in data_copy.index:
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
    data_copy['area_type'] = [get_type(name, marking) for name in
                              data_copy.index]  # data_copy['name'].apply(lambda s: get_type(s, marking))

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
        name = s.name

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

        return match.index[0]

    data_copy.index = data_copy.apply(match_to_krs, axis=1)

    geo_data_krs = geo_data_krs.join(data_copy, rsuffix='_data')

    geo_data_krs = geo_data_krs.drop(['area_type', 'score'], axis=1)

    geo_data_krs['has_data'] = geo_data_krs['value'].notna()

    geo_data_krs['centroids'] = geo_data_krs.centroid



    ax.set_title(title, y=1.05, fontdict={'fontweight':'bold'})

    '''Plot the data'''
    if vmin is None:
        vmin = geo_data_krs[geo_data_krs['has_data']].min().min()
    if vmax is None:
        vmax = geo_data_krs[geo_data_krs['has_data']].max().max()

    ax = geo_data_krs[geo_data_krs['has_data']].plot(ax=ax, column='value', linewidth=0, zorder=3, legend=False, vmin=vmin, vmax=vmax, cmap=cmap)

    if own_fig:
        cax = fig.add_axes([1, 0.1, 0.03, 0.8])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbr = fig.colorbar(sm, cax=cax, label=title_axis)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

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

    if crop:
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

    if save_to is not None:
        # Create the output folder, if it does not exist
        save_to.replace('\\', '/')
        save_figure(fig, save_to, format)
    return fig, ax



def process_data(df: pd.DataFrame, n_rows=1, n_cols=1, zoom=1, figure_title:str='', **kwargs) -> None:
    vmin = df.min().min()
    vmax = df.max().max()
    individual_files = True
    fig=None
    ax=None

    if n_rows != 1 or n_cols != 1:
        if len(df) != n_rows*n_cols:
            raise Exception(f'The dimensions of the subplots of {n_rows}x{n_cols} have been passed, but there are {len(df)} rows in the data!')
        individual_files = False

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))


    for index, row in df.iterrows():
        if not individual_files:
            ax = axes.flatten()[index]

        tmp = pd.DataFrame(row)
        tmp = tmp.rename(columns={index: 'value'})
        args = kwargs.copy()
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
        plot_geo_data(tmp, fig=fig, ax=ax, vmin=vmin, vmax=vmax, zoom=zoom, **args)


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
    y0_colorbar = max_y1 - (n_rows*(ax_height + y_gap_width) - y_gap_width)
    y1_colorbar = (n_rows*(ax_height + y_gap_width) - y_gap_width)

    cax = fig.add_axes([x0_colorbar, y0_colorbar, 0.03, y1_colorbar])
    sm = plt.cm.ScalarMappable(cmap=kwargs['cmap'], norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbr = fig.colorbar(sm, cax=cax, label=kwargs['title_axis'])

    # add the figure title
    if figure_title:
        fig.suptitle(figure_title, y=0.95, fontweight='bold', fontsize=16)

    # save to a file
    if not individual_files and kwargs['save_to'] is not None:
        save_figure(fig, kwargs['save_to'], kwargs['format'])

    return fig, axes

def process_file(filename: str, **kwargs) -> None:
    df = pd.read_csv(filename)
    print('The following data has been read:')
    print(df)
    process_data(df, **kwargs)

def main(argv):
    _args = dict(
        filename='ExampleData.csv',
        title_axis='Please add a title to the colorbar!',
        title='Random example data',
        cmap='RdYlGn',
        text_alpha=0.5,
        label_counties=True,
        crop=True,
        add_roads=False,
        use_cx=False,
        save_to=None,
        format='png'
    )

    try:
        opts, args = getopt.getopt(argv, '"hf:a:t:o:',
                                   ['filename=', 'title_axis=', 'title=', 'cmap=', 'text_alpha=', 'save_to=', 'format=', 'label_counties', 'add_roads', 'use_cx'])
    except getopt.GetoptError:
        print(__doc__)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(__doc__)
            sys.exit()
        elif opt in ("-i", "--filename"):
            _args['filename'] = arg
        elif opt in ("-a", "--title_axis"):
            _args['title_axis'] = arg
        elif opt in ("-t", "--title"):
            _args['title'] = arg
        elif opt in ("--cmap"):
            _args['cmap'] = arg
        elif opt in ("--format"):
            _args['format'] = arg
        elif opt in ("-o", "--save_to"):
            _args['save_to'] = arg
        elif opt in ("--text_alpha"):
            _args['text_alpha'] = float(arg)
        elif opt in ("--label_counties"):
            _args['label_counties'] = True
        elif opt in ("--add_roads"):
            _args['add_roads'] = True
        elif opt in ("--use_cx"):
            _args['use_cx'] = True

    if _args['save_to'] is None:
        _args['save_to'] = f'output/{_args["filename"][:_args["filename"].rfind(".")]}'

    print(f'The data file is {_args["filename"]}')
    print(f'The title is {_args["title"]}')
    print(f'The axis title is {_args["title_axis"]}')
    print(f'The colormap is {_args["cmap"]}')

    print(f'\nPlotting data from the csv file and saving each line to an individual {_args["format"]}-file')
    process_data(**_args)
    print('All saved.')


if __name__ == '__main__':
    main(sys.argv[1:])
