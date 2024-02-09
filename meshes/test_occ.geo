// Specify that OpenCASCADE should be used
SetFactory("OpenCASCADE");

// Define the circle
Circle(1) = {0, 0, 0, 1.0, 0, 2*Pi};
Circle(2) = {0, 0, 0, 0.9, 0, 2*Pi}; // Smaller circle to use as a tool

// Define the triangle
Point(3) = {0, 0, 0, 1.0}; // Vertex at the center of the circle
Point(4) = {0.2, 0.1, 0, 1.0};
Point(5) = {0.4, 0, 0, 1.0};
Line(6) = {4, 5};
Line(7) = {5, 3};
Line(8) = {3, 4};
Line Loop(9) = {6, 7, 8};
Plane Surface(10) = {9};

// Perform the Boolean subtraction
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }

// Set the physical group for the resulting surface
Physical Surface(11) = {10};
