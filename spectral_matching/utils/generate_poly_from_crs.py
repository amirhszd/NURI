from shapely.geometry import Polygon
from pyproj import Proj, Transformer
from shapely.ops import cascaded_union, unary_union
import numpy as np


def poly_from_latlon(polygon, transform):
    """
    Auxilary function grabbing polygons and transforming from any coordinates to UTM coordinates
    to image coordinates.
    """
    poly_pts = []
    transformer = Transformer.from_crs("epsg:4326", transform.crs, always_xy=True)

    poly = unary_union(polygon)
    for i in np.array(poly.exterior.coords):
        utm_x, utm_y = transformer.transform(i[0], i[1])
        poly_pts.append(~transform * (utm_x, utm_y))

    new_poly = Polygon(poly_pts)
    return new_poly


def poly_from_utm(polygon, transform):
    """
    Auxilary function grabbing polygons and transforming from any coordinates to UTM coordinates
    to image coordinates.
    """
    poly_pts = []

    poly = unary_union(polygon)
    for i in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(i))

    new_poly = Polygon(poly_pts)
    return new_poly
