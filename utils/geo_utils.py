import pandas as pd
import pgeocode
from geopy.distance import geodesic

_nom = pgeocode.Nominatim('IN')  # India

def compute_distance(zip1: str, zip2: str) -> float:
    """
    Returns distance in kilometers between two Indian postal codes.
    Falls back to a large number if lookup fails.
    """
    loc1 = _nom.query_postal_code(zip1)
    loc2 = _nom.query_postal_code(zip2)
    if pd.isna(loc1.latitude) or pd.isna(loc2.latitude):
        return 9999.0
    return geodesic(
        (loc1.latitude, loc1.longitude),
        (loc2.latitude, loc2.longitude)
    ).km
