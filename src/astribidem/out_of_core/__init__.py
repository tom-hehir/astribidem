"""Match catalogues larger than memory: band by band, then segment by segment.

``banded`` builds candidate edges one declination band at a time from a
streamed layout, using the region engine in ``regions``; ``index_files``
resolves saved edges one segment at a time into a single index file. The
public functions are exported from the package root.
"""
