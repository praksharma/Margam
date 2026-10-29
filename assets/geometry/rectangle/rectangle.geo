// Rectangular outline from Gmsh tutorial t1: x=0..0.1, y=0..0.3, z=0.
// Source: https://gmsh.info/doc/texinfo/gmsh.html#t1
// Standalone OpenCASCADE construction for CAD import testing.
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 0.1, 0.3};
