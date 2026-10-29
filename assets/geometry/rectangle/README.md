# Rectangle for Jorvik import testing

Use **rectangle.geo** for the current planar sampling workflow. It is a single
0.1 × 0.3 rectangle in the XY plane, with one face and four boundary edges.
Jorvik import and 32 Sobol interior samples were checked.

**rectangle.step** contains the same outline as a CAD face. It opens in Jorvik,
but the current Jorvik implementation classifies STEP sheets as open/nonplanar
and does not enable planar interior sampling for them. Use the GEO file until
that upstream restriction is extended.

The outline comes from [Gmsh's official t1 tutorial](https://gmsh.info/doc/texinfo/gmsh.html#t1).
These are independent standalone constructions of its geometric coordinates,
not an original STEP download or a copy of the tutorial's implementation.
`source.json` records provenance and import checks. Coordinates have no assigned
physical units in the GEO definition.

In Margam, open the Jorvik geometry block's editor, then import `rectangle.geo`.
Select its four edges to create boundary regions, sample the face interior and
boundaries, and export `domain.nc` beside the owned `domain.jorvik` document.
