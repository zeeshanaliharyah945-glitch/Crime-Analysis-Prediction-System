# map_utils.py
# =====================================================================
#  Shared map helpers for the Crime Analysis & Prediction System
# =====================================================================
#  Why this file exists
#  --------------------
#  CARTO started requiring an API key for basemaps.cartocdn.com in
#  late August 2026. Maps that still use the old "CartoDB positron" /
#  "CartoDB dark_matter" tiles now show an "API KEY REQUIRED" watermark.
#
#  Every map in this project is now created through create_map(), which
#  uses key-free tile providers (Esri + OpenStreetMap) and lets the user
#  switch the basemap from the layer control (top-right corner of the map).
#
#  Optional: if you own a free CARTO key you can enable the CARTO layers by
#  setting an environment variable before running the project:
#       Windows (cmd):         set CARTO_API_KEY=your_key_here
#       Windows (PowerShell):  $env:CARTO_API_KEY="your_key_here"
#       Linux / macOS:         export CARTO_API_KEY=your_key_here
#  Without the variable, no CARTO tiles are requested at all.
# =====================================================================

import functools
import http.server
import os
import socketserver
import threading
import webbrowser

import folium

ESRI_ATTR = ('Tiles &copy; Esri &mdash; Esri, HERE, Garmin, FAO, NOAA, USGS, '
             '&copy; OpenStreetMap contributors')
OSM_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
CARTO_ATTR = ('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
              'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>')

_ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/{name}/MapServer/tile/{{z}}/{{y}}/{{x}}'

# name -> (url template, attribution)
BASE_LAYERS = {
    'Light':      (_ESRI.format(name='Canvas/World_Light_Gray_Base'), ESRI_ATTR),
    'Dark':       (_ESRI.format(name='Canvas/World_Dark_Gray_Base'), ESRI_ATTR),
    'Streets':    (_ESRI.format(name='World_Street_Map'), ESRI_ATTR),
    'Satellite':  (_ESRI.format(name='World_Imagery'), ESRI_ATTR),
    'OpenStreetMap': ('https://tile.openstreetmap.org/{z}/{x}/{y}.png', OSM_ATTR),
}

# Label-only overlays so the gray base maps still show street / place names.
_REFERENCE = {
    'Light': _ESRI.format(name='Canvas/World_Light_Gray_Reference'),
    'Dark':  _ESRI.format(name='Canvas/World_Dark_Gray_Reference'),
}

_THEME_TO_LAYER = {'light': 'Light', 'dark': 'Dark', 'streets': 'Streets',
                   'satellite': 'Satellite', 'osm': 'OpenStreetMap'}


def _carto_key():
    return os.environ.get('CARTO_API_KEY', '').strip()


def create_map(location, zoom_start=13, theme='light', **kwargs):
    """Create a folium map with key-free basemaps.

    theme: 'light', 'dark', 'streets', 'satellite' or 'osm' (the basemap shown first).
    All other basemaps stay available in the layer control. Remember to add
    folium.LayerControl().add_to(map) after adding your own layers.
    """
    default = _THEME_TO_LAYER.get(str(theme).lower(), 'Light')
    m = folium.Map(location=location, zoom_start=zoom_start, tiles=None, **kwargs)

    layers = dict(BASE_LAYERS)
    key = _carto_key()
    if key:
        base = 'https://{s}.basemaps.cartocdn.com/rastertiles/'
        layers['CARTO Light'] = (base + 'light_all/{z}/{x}/{y}.png?key=' + key, CARTO_ATTR)
        layers['CARTO Dark'] = (base + 'dark_all/{z}/{x}/{y}.png?key=' + key, CARTO_ATTR)

    # default basemap first, then the rest
    order = [default] + [n for n in layers if n != default]
    for name in order:
        url, attr = layers[name]
        native = 16 if name in _REFERENCE else 19      # Esri gray canvas maps stop at zoom 16
        folium.TileLayer(tiles=url, attr=attr, name=name, overlay=False, control=True,
                         show=(name == default), max_zoom=19, max_native_zoom=native).add_to(m)

    # place / road labels on top of the gray Esri maps
    for name, url in _REFERENCE.items():
        folium.TileLayer(tiles=url, attr=ESRI_ATTR, name=f'{name} labels', overlay=True,
                         control=False, show=(name == default), max_zoom=19,
                         max_native_zoom=16).add_to(m)
    return m


# ---------------------------------------------------------------------
#  Opening maps in the browser
# ---------------------------------------------------------------------
#  OpenStreetMap blocks tile requests that carry no HTTP Referer, and a page
#  opened with file:// never sends one. Serving the map through a tiny local
#  web server (http://127.0.0.1:<port>/) fixes that, so all basemaps work.
# ---------------------------------------------------------------------
_servers = {}


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _serve_directory(directory):
    directory = os.path.abspath(directory)
    if directory not in _servers:
        handler = functools.partial(_QuietHandler, directory=directory)
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        _servers[directory] = server
    return _servers[directory].server_address[1]


def open_map(path, serve=True):
    """Open an HTML map in the default browser.

    serve=True  : use a local web server (needs the program to keep running,
                  which is the case for the GUI) - every basemap works.
    serve=False : plain file:// URL (for scripts that exit right away); the
                  default Esri basemaps work, OpenStreetMap may be blocked.
    """
    path = os.path.abspath(path)
    if serve:
        try:
            port = _serve_directory(os.path.dirname(path))
            webbrowser.open(f'http://127.0.0.1:{port}/{os.path.basename(path)}')
            return
        except OSError:
            pass  # fall back to file://
    webbrowser.open('file:///' + path.replace('\\', '/'))
