Point (0) = { 0, 0, 0, 0.025 };
Point (1) = { -0.9238795325112867, 0.3826834323650898, 0.0, 0.1 };
Point (2) = { -0.9238795325112867, -0.3826834323650898, 0.0, 0.1 };
Point (12) = { 1.0, 0, 0.0, 0.025 };
Point (10) = { -1.1086554390135441, 0.4592201188381077, 0.0, 0.1 };
Point (20) = { -1.1086554390135441, -0.4592201188381077, 0.0, 0.1 };
Point (120) = { 1.2, 0, 0.0, 0.1 };

Line (3) = { 1, 0 };
Line (4) = { 0, 2 };

Circle (5) = { 2, 0, 12 };
Circle (6) = { 12, 0, 1 };

Line (30) = { 10, 1 };
Line (40) = { 2, 20 };

Circle (50) = { 1, 0, 12 };
Circle (60) = { 12, 0, 2 };
Circle (51) = { 20, 0, 120 };
Circle (61) = { 120, 0, 10 };

Curve Loop(1000) = { 3, 4, 5, 6 };
Curve Loop(1010) = { 30, 50, 60, 40, 51, 61 };

Plane Surface (100) = { 1000 };
Plane Surface (101) = { 1010 };

Physical Surface(200) = { 100 };
Physical Surface(201) = { 101 };