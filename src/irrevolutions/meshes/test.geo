// Define points
Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};
Point(5) = {1, 0, 0, 1.0};
Point(6) = {2, 0, 0, 1.0};
Point(7) = {2, 1, 0, 1.0};
Point(8) = {1, 1, 0, 1.0};

// Define rectangles
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

// Define the common boundary
Physical Line(9) = {1, 5};  // Assign the same physical group to both sides

// Define surfaces
Line Loop(10) = {1, 2, 3, 4};
Plane Surface(11) = {10};
Line Loop(12) = {5, 6, 7, 8};
Plane Surface(13) = {12};

// Set the physical groups for the entire domain
Physical Surface(14) = {11, 13};
