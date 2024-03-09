L = 50.;

f0 = 0.2/L;

eta = 0.01/L;

xval = ;
yval = 0;
xtip = 0;
tip = 0.3/L;



//+
Point(1) = {xval, yval+eta, 0, f0};
//+
Point(2) = {xval, yval-eta, 0, f0};
//+
Point(3) = {xtip-tip, yval+eta, 0, f0};
//+
Point(4) = {xtip-tip, yval-eta, 0, f0};
//+
Point(5) = {xtip,yval,0,f0};



//+
Line(1) = {1, 3};
//+
Line(2) = {3, 5};
//+
Line(3) = {5, 4};
//+
Line(4) = {4, 2};
