close all
clear all
load('CV_valid.mat')
lateral = CV_valid(1,:)';
radial = CV_valid(2,:)';

% Visual estimation of angles:
g1_zero = [36;-771];
g2_zero = [761;15];

g1 = [-425,-206, -16,33;
    -629,-729,-756,-757];
g2 = [620,727,760,763;
    -443,-225,-31,19];


g1_zero_angle = atan2(g1_zero(2,1),g1_zero(1,1));
g2_zero_angle = atan2(g2_zero(2,1),g2_zero(1,1));

for i = 1:length(g1)
    g1_ang(i) = (atan2(g1(2,i),g1(1,i)) - g1_zero_angle);
    g1_mm(i) = -(g1_ang(i) * 10/(2*pi))';
    g2_ang(i) = (atan2(g2(2,i),g2(1,i)) - g2_zero_angle);
    g2_mm(i) = -(g2_ang(i) * 10/(2*pi));
end
g1_mm=g1_mm';
g2_mm =  g2_mm';

% resolution of the measurements, in mm
resolution = sin(1/740)*10
% max difference between visual estimation and computer vision algorithm:
maxLateral = max(abs(lateral-g1_mm))
maxRadial = max(abs(radial-g2_mm))

measNumber = [1;2;3;4];
figure()
subplot(2,1,1)
hold on
plot(measNumber,lateral, 'x','MarkerSize',10)
plot(g1_mm,'d','MarkerSize',10)
legend('CV','visual')
xlabel('Measurement Number')
ylabel('Displacement [mm]')
title('Gauge CV Validation')
ax = gca; % current axes
ax.FontSize = 12;
xticks([1 2 3 4])
subplot(2,1,2)
hold on
plot(measNumber,radial, 'x','MarkerSize',10)
plot(g2_mm,'d','MarkerSize',10)
legend('CV','visual')
xlabel('Measurement Number')
ylabel('Displacement [mm]')
hold off
ax = gca; % current axes
ax.FontSize = 12;
xticks([1 2 3 4])





