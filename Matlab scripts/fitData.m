function [A,a_coef] = fitData(disp,spoke)
%m measurements
m = length(disp);
theta0 = (2*spoke -1)*(2*pi/m);
theta=(1:m)*(2*pi/m) - theta0;
A = zeros(m,m-1);
% first entry is a constant term (n=0 --> cos ntheta = 1)
A(:,1) = 1;
% subsequent entries are cos ntheta sin ntheta terms
for row = 1:m
    for col = 2:m-1
        n = floor(col/2);
        if rem(col,2) == 0
            A(row,col) = cos(theta(row)*n);
        else 
            A(row,col) = sin(theta(row)*n);
        end
    end
end
a_coef = A\disp;
end