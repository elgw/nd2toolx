#!/bin/env python

"""For debugging, use command like
python -m pdb -c continue ./nd2tool_extra.py --nd2 /srv/backup/FFPE_FISH/iiXZ0215_20231221_008.nd2 --out 215

# Changelog

20250506:

- New: Creates a layout.svg.log.txt besides layout.svg with the
coordinate conversion (pixels to microscopy)

- New: Writes out the coordinate conversion formula also for tiled
  images.

Intial version 20250501, Erik Wernersson.

"""
import argparse
import csv
import glob
import io
import math
import sys
import os
import subprocess
import json
import warnings

# External dependencies
import cairo
import pandas as pd
from tifffile import imread, imwrite
import numpy as np
from PIL import Image # pillow
from scipy.ndimage import gaussian_filter

# Not given as a warning...
import warnings
warnings.filterwarnings("ignore", message=".*ImageJ.*")


def parse_command_line() -> argparse.Namespace:
    """ Parse the command line arguments and perform some initial checks """

    progdesc = """
    This is a companion script to nd2tool which can be
    used to figure out where individual FOV are located on the slide
    as well as tile individual channels and merge them to a 2D composite
    image

    For usage, please see the help sections for the respective sub commands.

    """

    parser = argparse.ArgumentParser(
        description=progdesc,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparser = parser.add_subparsers(help="Commands", dest='command')

    ## LAYOUT
    layoutdesc="""Create an SVG image which shows the layout of the FOV"""
    layout = subparser.add_parser("layout",
                                  help="Generate an SVG image showing the layout of the FOV",
                                  formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=layoutdesc)

    layout.add_argument('--outdir', required=False,
                        help='where to put generated files')
    layout.add_argument('--nd2', required=True,
                        help='nd2 file to use')

    ## TILE
    tiledesc = """
    Takes and nd2 file as input and generates a tiled png file, and possibly even a dzi file.
    The script is run twice, the first time two configuration files are generated, please edit
    them to define the contrast and color of the individual channels. It is also possibel to remove
    specific FOV and to move them around.

    Example usage:
    # First run, generates configuration files
    $ ./nd2tool_extra.py --nd2 /the/server/iRB0156_20200217_001.nd2  --outdir iRB156 --dzi
    # set colors
    $ emacs iRB156/tile_channels.json
    # possibly exclude FOV etc
    $ emacs iRB156/tile_geometry.csv
    # Then run again to generate the output files
    $ ./nd2tool_extra.py --nd2 /the/server/iRB0156_20200217_001.nd2  --outdir iRB156 --dzi
    """
    tile = subparser.add_parser("tile", help="Tile FOV to single image",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=tiledesc)
    tile.add_argument('--nd2', required=True,
                        help='nd2 file to use')

    # NOTE: nd2tool always writes the output files in a subdirectory of the current dir

    tile.add_argument('--outdir', help="""where to place the
    configuration files and the output from this script""")

    ## DZI
    dzidesc="""Starting with a (huge) png image, this creates a tiled
    DZI image that can be rendered with OpenSeadragon etc."""

    dzi = subparser.add_parser("dzi", help="Create a DZI image from a png file",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=dzidesc)
    dzi.add_argument('--png', help="Input file to use (.png)", required=True)

    dzi.add_argument('--out', help='Name of the output file', required=False)

    config = parser.parse_args()

    return config

def validate_tile_arguments(config):
    if not os.path.exists(config.nd2):
        print(f"{config.nd2} does not exist", file=stderr)
        sys.exit(1)

    # This is all we can do at the moment since nd2tool can't place the files in any other folder
    config.imdir = os.getcwd() + os.sep + os.path.split(config.nd2)[-1].replace('.nd2', os.sep)

    if config.imdir is None:
        config.imdir = config.nd2.replace('.nd2', os.sep)

    if config.outdir is None:
        config.outdir = config.imdir

    if not os.path.isdir(config.outdir):
        print(f"Creating {config.outdir}")
        os.mkdir(config.outdir)

    config.channel_file = config.outdir + os.sep + 'tile_channels.json'
    config.geometry_file = config.outdir + os.sep + 'tile_geometry.csv'

    print(f"nd2 file: {config.nd2}")
    print(f"imdir: {config.imdir}")
    print(f"channel file: {config.channel_file}")
    print(f"geometry file: {config.geometry_file}")

def get_coords(nd2file : str) -> 'pandas.dataframe':
    """ run
    $ nd2tool --coord "${nd2file}"
    which produces 6 col csv
    return a pandas datafram
    """

    cmd = ['nd2tool', '--coord', nd2file]
    try:
        output = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting coordinates from {config.nd2}")
        sys.exit(1)

    cmdout = io.StringIO(output.stdout.decode())
    df = pd.read_csv(cmdout, sep=",")
    df.columns = [x.strip() for x in df.columns]
    # df.to_csv(coords_file, sep=",")
    return df


def get_metadata(config) -> list:
    """
    Sets dx_um and tile_size_px in config, returns a list with per channel info
    the information comes from

    $ nd2tool --meta-file "${nd2file}" > meta-file.json

    """


    nd2file = config.nd2
    cmd = ['nd2tool', '--meta-file', nd2file]
    try:
        output = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting metadata from {config.nd2}")
        sys.exit(1)

    jmeta = json.loads(output.stdout.decode())

    meta = []

    config.dx_um = jmeta['channels'][0]['volume']['axesCalibration'][0]

    config.tile_size_px = jmeta['channels'][0]['volume']['voxelCount'][0]

    for c in jmeta['channels']:
        chan = {}
        chan['name'] =  c['channel']['name']
        meta.append(chan)

    return meta


def select_value(A, k):
    # Quickselect
    if( k > A.size/2):
        vals = np.partition(A, k)
        vals = vals[k:]
        vals.sort()
        return vals[0]
    else:
        vals = np.partition(A, k)
        vals = vals[:k]
        vals.sort()
        return vals[-1]


def create_composite(config, chanconf):
    # python ../create_composite.py comp.json

    outfile = config.outdir + os.path.sep + 'composite.png'

    if os.path.isfile(outfile):
        print(f"{outfile} does already exist")
        return


    C = None
    for channel in chanconf:
        print(f"Processing {channel['name']}")
        print(".", end="", flush=True)
        infile = config.imdir + 'tiled_' + channel['name'] + '.tif'
        if not os.path.isfile(infile):
            print(f"{infile} does not exist, skipping it")
            continue

        I = imread(infile)

        # I = I.astype('np.float32') ??
        # I = I[0:2*2048, 0:2*2048]
        assert(len(I.shape) == 2) # Only 2D

        if C is None:
            C = np.zeros( (3, I.shape[0], I.shape[1]), dtype=np.float32) #, dtype=np.uint8)

        # scale between 0 and 1

        if 0:
            I = I - float(channel['range'][0])
            I = I / float( (channel['range'][1]-channel['range'][0]) )
        # I = 0*I + 1;

        print(".", end="", flush=True)
        # Percentiles
        I0 = I.flatten()
        I0 = I0[I0 > 0]
        low_percentile = channel["percentile"][0]
        high_percentile = channel["percentile"][1]
        print(f"Percentiles: {low_percentile}, {high_percentile}")
        print(f"Pixel values: {np.min(I.flatten())}, {np.max(I.flatten())}")
        high = select_value(I0, int(high_percentile*I0.size))
        low = select_value(I0, int(low_percentile*I0.size))
        del I0

        print(f"Range: {low} -- {high} from select_value")

        if 0: # Gives the same answer but slower
            idx = np.argsort(I.flatten())
            low = I.flatten()[idx[int(low_percentile*I.size)]]
            high = I.flatten()[idx[int(high_percentile*I.size)]]
            del idx
            print(f"Range: {low} -- {high}")

        #breakpoint()
        print(".", end="", flush=True)
        I = I - low

        I[I < 0.0] = 0.0;
        I = I/(high-low)
        I[I > 1.0] = 1.0

        # convert to 8u
        # I = I.astype(np.uint8)
        r = float(channel['color'][0])
        g = float(channel['color'][1])
        b = float(channel['color'][2])
        if r > 0:
            C[0, :, :] = C[0, :, :] + r*I
        if g > 0:
            C[1, :, :] = C[1, :, :] + g*I
        if b > 0:
            C[2, :, :] = C[2, :, :] + b*I
        print(".")

    print("Converting to RGB")
    #breakpoint()

    # C2 = C[:, 0:2*2048, 0:2*2048]

    C = C.astype(np.uint8)
    C = C.swapaxes(0, 2)
    C = C.swapaxes(0, 1)

    im = Image.fromarray(C, mode='RGB')

    print(f"Writing to {outfile}")
    im.save(outfile)

    return


def parse_dw_scaling(filename):
    if not "dw_" in filename:
        return 1.0

    logfile = filename + ".log.txt"
    logfile = logfile.replace('max_dw', 'dw')

    with open(logfile, "r") as fid:
        for line in fid:
            if "scaling" in line:
                parts = line.split(' ')
                return float(parts[1])

    print(f"Unable to parse scaling for {filename} in {logfile}")
    print("Due to --tilesize?")
    return 1.0

def downscale_image(image, factor):
    """ Downscale an image by an integer factor using
    averaging """
    if(factor == 1):
        return image

    image = image[0:-1, 0:-1]
    + image[1:, 0:-1]
    + image[0:-1, 1:]
    + image[1:, 1:]

    image = image[0::2, 0::2]

    return downscale_image(image, factor / 2)

def gsmooth(im, sigma):
    """ Normalized Gaussian smoothing """
    a = gaussian_filter(im, sigma)
    b = gaussian_filter(0*im+1, sigma)
    return a / b


def estimate_bg(folder, shortname):
    savename = os.path.join(folder, 'bg_' + shortname + '.tif')
    print(f"savename: {savename}")
    if os.path.isfile(savename):
        bg = imread(savename)
        bg = bg.astype(np.float32)
        return bg

    spattern = os.path.join(folder, shortname) + '*.tif';
    print(f"Search pattern: {spattern}")
    files = glob.glob(spattern);
    files.sort()
    print(f"Found {len(files)}")

    #breakpoint()
    assert(len(files) > 0)

    if len(files) < 10:
        print(f"Only {len(files)} available, background estimation disabled")
        test = imread(files[0])
        bg = np.ones((test.shape[-2], test.shape[-1]))
        return bg

    bg = imread(files[0])
    bg = np.zeros((bg.shape[-1], bg.shape[-1]))

    #with ProgressBar(max_value=len(files)) as bar:
    for i, file in zip(range(0, len(files)), files):
        t = imread(file)
        if len(t.shape) != 2:
            print(f"The shape of {file} is {t.shape} which is not ok")
            sys.exit(1)

        if len(t.shape) > 2:
            t = t.sum(axis=0)
        bg = bg + t
        # print(bg.shape)
        # breakpoint()
        #bar.update(i)

    assert(len(bg.shape) == 2)
    bg = bg.astype(np.float32)

    bg = gsmooth(bg, 100)
    bg = bg / np.max(bg)

    print(f"Writing background estimate to {savename}")
    imwrite(savename, bg)
    return bg


def tile_channel(cname, config, meta, geo):
    # estimate bg, returns a constant image if too few images available

    outfile = config.imdir + os.path.sep + 'tiled_' + cname + '.tif'

    if os.path.isfile(outfile):
        print(f"{outfile} already exists, tiling done")
        return

    geometry = geo.copy()
    geometry['temp'] = geometry['x_px']
    geometry['x_px'] = geometry['y_px']
    geometry['y_px'] = geometry['temp']
    if True: # invert
        geometry['y_px'] = -geometry['y_px']

    minx = np.min(geometry['x_px'])
    miny = np.min(geometry['y_px'])

    geometry['x_px'] = geometry['x_px'] - minx
    geometry['y_px'] = geometry['y_px'] - miny

    with open(outfile + '.log.txt', 'w') as fid:
        xa = (geometry['X_um'][1] - geometry['X_um'][0]) / (geometry['y_px'][1] - geometry['y_px'][0])
        xb = geometry['X_um'][0] - xa*geometry['y_px'][0]
        ya = (geometry['Y_um'][1]-geometry['Y_um'][0]) / (geometry['x_px'][1]-geometry['x_px'][0])
        yb = geometry['Y_um'][0] - ya*geometry['x_px'][0]

        fid.write(f"""
        Please note that the x and y are inverted in the
        tiled image compared to the microscopy coordinate system\n""")
        fid.write(f"To get the microscopy coordinates (u, v) use:\n");
        fid.write(f"u = ({xa})*y + ({xb})\n")
        fid.write(f"v = ({ya})*x + ({yb})\n")
        fid.write(f"Where (x,y) are the pixel coordinates\n");

        # xa*geometry['y_px']+xb - geometry['X_um']
        # ya*geometry['x_px']+yb - geometry['Y_um']
        assert(np.max(xa*geometry['y_px']+xb - geometry['X_um']) < 1e-6)
        assert(np.max(ya*geometry['x_px']+yb - geometry['Y_um']) < 1e-6)

    width = np.max(geometry['x_px']) + config.tile_size_px + 1
    height = np.max(geometry['y_px']) + config.tile_size_px + 1
    width = width.astype('int')
    height = height.astype('int')

    print(f"Output size: {width} x {height}")
    im = np.zeros((width, height), dtype=np.float32)
    bg = estimate_bg(config.imdir, 'max_' + cname)

    files = glob.glob(config.imdir + os.path.sep + f'max_{cname}*.tif')
    files.sort()


    coords = np.array( (geometry['x_px'], geometry['x_px'] + config.tile_size_px, geometry['y_px'], geometry['y_px'] + config.tile_size_px))
    coords = coords.transpose()
    downscale = 1 # TODO as command line argument

    for filename, pos in zip(files, coords):
        # print(f"{filename}")
        tile = imread(filename)
        tile = tile.astype(np.float32)
        tile_scaling = parse_dw_scaling(filename)
        tile = tile / tile_scaling

        if len(tile.shape) == 3:
            tile = np.max(tile, axis=0)
        #tile = downscale_image(tile, args.downscale)
        tile = tile / bg
        x0 = round(pos[0]/downscale)
        x1 = round(pos[1]/downscale)
        y0 = round(pos[2]/downscale)
        y1 = round(pos[3]/downscale)
        print(f"{x1-x0}, {y1-y0}")
        print(tile.shape)

        im[x0:x1, y0:y1] = tile


    print(f"Writing to {outfile}")
    imwrite(outfile, data=im, compression=None)


def tile_channels(config, meta, geometry):
    """ Tile the channels once by one """

    for channel in meta:
        tile_channel(channel['name'], config, meta, geometry)

    return

def to_tiff(nd2 : str) -> None:
    print(f"Converting nd2 to tif")
    cmd = ['nd2tool', nd2]
    try:
        output = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error converting to tiff, file: {nd2}")
        sys.exit(1)
    print(f"{output.stdout.decode()}")
    return

def load_channel_file(channel_file):
    with open(channel_file, 'rb') as f:
        cdata = json.load(f)
    return cdata

def gen_channel_file(tilespec_file, meta):
    """
    Example output:
    [
    {
        "file":"max_dw_dapi_merged.tif",
        "percentile" : [0.01, 0.99],
        "color": [0, 0, 255]
    },
    {
        "file":"max_dw_A594_merged.tif",
        "percentile" : [0.10, 0.99],
        "color": [255, 0, 0]
    },
    {
        "file":"max_dw_A647_merged.tif",
        "percentile" : [0.10, 0.99],
        "color": [0, 255, 0]
    }
    ]

    """
    if os.path.isfile(tilespec_file):
        print(f"{tilespec_file} does already exist")
        sys.exit(1)

    tilespec = []
    for c in meta:
        conf = {}
        conf['name'] = c['name']
        conf['percentile'] = [0.1, 0.99]
        conf['color'] = [0, 0, 255]
        tilespec.append(conf)

    # dict
    y = json.dumps(tilespec, indent=4)

    print(f"Writing tilespec template to {tilespec_file}")
    with open(tilespec_file, "wb") as fid:
        fid.write(y.encode('utf8'))

    return

def maxproj_image(file : str) -> None:

    cmd = ['dw', 'maxproj', file]
    try:
        output = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        breakpoint()
        print(f"Error creating max projections for: file")
        sys.exit(1)


def maxproj_images(nd2 : str, meta : list) -> None:
    print(f"Creating max projections")
    # figure out what folder to run in
    # run dw maxproj
    [nd2dir, nd2file] = os.path.split(nd2)
    folder = nd2file.replace('.nd2', os.sep)
    # TODO: wildcards not expanded ... per file ...
    for c in meta:
        files = glob.glob(f'{folder}{c["name"]}*.tif')
        files.sort()
        for fname in files:
            maxproj_image(fname)


def gen_geometry(config):
    print(f"Generating geometry file: {config.geometry_file}")

    Coords = get_coords(config.nd2)
    Coords = Coords.loc[Coords['Channel'] == 1, :]
    Coords = Coords.loc[Coords['Z'] == 1, :]
    Coords['x_px'] = Coords['X_um'] / config.dx_um
    Coords['y_px'] = Coords['Y_um'] / config.dx_um
    Coords['x_px'] = Coords['x_px'] - np.min(Coords['x_px'])
    Coords['y_px'] = Coords['y_px'] - np.min(Coords['y_px'])

    width = np.max(Coords['x_px']) + config.tile_size_px
    height = np.max(Coords['y_px']) + config.tile_size_px
    print(f"Output size: {int(width)} x {int(height)} pixels")

    Coords.to_csv(config.geometry_file, sep=',')
    return

def get_geometry(config) -> 'pandas.dataframe':
    geo = pd.read_csv(config.geometry_file, sep=',')
    return geo

def gen_dzi(config):
    """
    Create a dzi file from the composite image. This can be viewed with openSeadragon
    """
    outname = config.outdir + os.path.sep + os.path.split(config.nd2)[-1].replace('.nd2', '')
    print(f"Creating {outname}.dzi")
    cmd = ['vips', 'dzsave', config.outdir + os.path.sep + 'composite.png', outname, '--suffix', '.png']
    try:
        output = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Errors generating dzi file {config.nd2}")
        print(f"Command: {' '.join(cmd)}")
        sys.exit(1)

def draw_boxes(ctx, coords, side):
    """ Side: Side length of squares """

    for row in coords.iterrows():

        x0 = row[1]['X']
        y0 = row[1]['Y']
        fov = row[1]['FOV']
        ctx.move_to(x0, y0)
        ctx.line_to(x0+side, y0)
        ctx.line_to(x0+side, y0+side)
        ctx.line_to(x0, y0+side)
        ctx.line_to(x0, y0);
        ctx.stroke()
        ctx.move_to(x0+side/3, y0+side/3)
        ctx.show_text(str(int(fov)))


def gen_svg_layout(config, geometry, chanconf):
    # from visualize_nd2_coords.py
    # Note: this inverts the coordinates

    svgname = config.outdir + os.path.sep + 'layout.svg'
    if os.path.isfile(svgname):
        print(f"{svgname} does already exist")

    title = config.nd2.split('.')
    title=title[-2]
    title=title.split('/')
    title = title[-1]
    print(f"Title: {title}")
    print(f'Reading info from {config.nd2}')


    coords = geometry.copy()
    # subselect only one row per fov
    # by taking the first Z and channel
    dx = config.dx_um
    imside = config.tile_size_px

    print(f"Lateral Pixel size: {dx} um")
    print(f"Image side length: {imside} pixels")

    # TODO: Write out the transformation from image pixels to microscope locations

    coords = coords[ coords['Z'] == 1 ]
    coords = coords[ coords['Channel'] == 1]
    print(f"Number of FOV: {len(coords)}")
    side = imside*dx;

    # Now we scale to fit a box of 1000 x 1000 pixels which will be output size of the svg image

    coords['X'] = - coords['X_um']
    minX = np.min(coords['X'])
    coords['X'] = coords['X'] - minX
    minY = np.min(coords['Y_um'])
    coords['Y'] = coords['Y_um'] - minY

    factor = max(1000/max(np.max(coords['X']) + side , np.max(coords['X']) + side),
                 1000/max(np.max(coords['Y']) + side , np.max(coords['Y']) + side));
    coords['X'] = coords['X'] * factor
    coords['Y'] = coords['Y'] * factor

    # coords['X'] = (-coords['X_um']-minX)*factor
    # coords['X_um'] = -coords['X']/factor-minX
    # a = -1/factor, b = -minX
    # coords['Y'] = (coords['Y_um']-minY)*factor
    # coords['Y_um'] = coords['Y']/factor + minY

    print(f"C(x) = {-1/factor}x + ({-minX})")
    print(f"C(y) =  {1/factor}y + ({minY})")

    #breakpoint()
    #  minX - coords['X']/factor
    side = side*factor

    xmax = np.max(coords['X']) + side;
    ymax = np.max(coords['Y']) + side;

    print(f'Writing to {svgname}')
    with cairo.SVGSurface(svgname, xmax, ymax) as surface:
        ctx = cairo.Context(surface)
        ctx.set_font_size(side/10)
        # Make background white, note transparent
        ctx.rectangle(0, 0, xmax, ymax)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)
        # Add a title
        ctx.move_to(1/10*side, 1/10*side)
        ctx.show_text(title)
        ctx.move_to(1/10*side, 1/10*side + 64)
        ctx.show_text(f"C(x) = {-1/factor}x + ({-minX})")
        ctx.move_to(1/10*side, 1/10*side + 128)
        ctx.show_text(f"C(y) =  {1/factor}y + ({minY})")
        draw_boxes(ctx, coords, side)

    with open(svgname+'.log.txt', 'w') as fid:
        fid.write(title + '\n')
        fid.write(f"C(x) = {-1/factor}x + ({-minX})\n")
        fid.write(f"C(y) =  {1/factor}y + ({minY})\n")

def layout(config):
    """ Generate an svg image with the layout of the tiles """

    validate_tile_arguments(config)

    meta = get_metadata(config)

    if not os.path.isfile(config.geometry_file):
        gen_geometry(config)

    geometry = get_geometry(config)

    if not os.path.isfile(config.channel_file):
        gen_channel_file(config.channel_file, meta)

    chanconf = load_channel_file(config.channel_file)

    gen_svg_layout(config, geometry, chanconf)
    return


def tile(config):
    validate_tile_arguments(config)

    meta = get_metadata(config)

    config_generated = False

    if os.path.isfile(config.geometry_file):
        geometry = get_geometry(config)
    else:
        config_generated = True
        gen_geometry(config)

    if os.path.isfile(config.channel_file):
        chanconf = load_channel_file(config.channel_file)
    else:
        gen_channel_file(config.channel_file, meta)
        config_generated = True

    if config_generated:
        print(f"Please edit the generated config files and run again")
        sys.exit(0)

    to_tiff(config.nd2)

    maxproj_images(config.nd2, meta)

    tile_channels(config, meta, geometry)

    create_composite(config, chanconf)


def cli():
    config = parse_command_line()

    if config.command == 'tile':
        tile(config)
    if config.command == 'layout':
        layout(config)
    if config.command == 'dzi':
        gen_dzi(config)

if __name__ == '__main__':
    cli()
