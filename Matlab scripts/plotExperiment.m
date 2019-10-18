function y = plotExperiment(figNum, Y_hat, Y,base, target_ten)
% takes the baseline data, predicted data, and post-truing data as single
% column vectors.  target tension is a scalar.
y=1;
theta_s = pi/32:pi/16:2*pi;
theta_r = pi/32:pi/32:2*pi;
% split data into components
Y_lat_hat = Y_hat(1:64);
Y_rad_hat = Y_hat(65:128);
Y_ten_hat = Y_hat(129:end);

Y_lat = Y(1:64);
Y_rad = Y(65:128);
Y_ten = Y(129:end);

base_lat = base(1:64);
base_rad = base(65:128);
base_ten = base(129:end);

figure(figNum)
subplot(3,1,1)
hold on
plot(theta_r,Y_lat_hat,'b-','LineWidth',1)
plot(theta_r,Y_lat,'kx','LineWidth',1)
hold off
ylabel('Lateral [mm]')
legend('Predict','Measure')
title('Prediction and Measurement After Truing Operation')
ax = gca;
ax.FontSize = 16;

subplot(3,1,2)
hold on
plot(theta_r,Y_rad_hat,'b-','LineWidth',1)
plot(theta_r,Y_rad,'kx','LineWidth',1)
hold off
ylabel('Radial [mm]')
legend('Predict','Measure')
ax = gca;
ax.FontSize = 16;
subplot(3,1,3)
data = cat(2,Y_ten_hat+target_ten,Y_ten);
bar(theta_s,data)
ylabel('Tension [N]')
xlabel('Rim Angle [rad]')
legend('Predict','Measure')
ax = gca;
ax.FontSize = 16;

figNum = figNum+1;
figure(figNum)
subplot(3,1,1)
plot(theta_r,Y_lat_hat- Y_lat,'b-','LineWidth',1)
ylabel('Lateral [mm]')
title('Residual Error After Truing')
ax = gca;
ax.FontSize = 16;

subplot(3,1,2)
plot(theta_r,Y_rad_hat- Y_rad,'b-','LineWidth',1)
ylabel('Radial [mm]')
ax = gca;
ax.FontSize = 16;

subplot(3,1,3)
bar(theta_s,Y_ten_hat - (Y_ten-target_ten))
ylabel('Tension [N]')
xlabel('Rim Angle [rad]')
ax = gca;
ax.FontSize = 16;

tension_m = cat(2,base_ten+target_ten,Y_ten);
figNum = figNum+1;
figure(figNum)
subplot(3,1,1)
hold on
plot(theta_r,base_lat,'LineWidth',1)
plot(theta_r,Y_lat,'LineWidth',1)
hold off
legend('pre','post')
ylabel('Lateral [mm]')
title('Measurements Before and After Truing Operation')
ax = gca;
ax.FontSize = 16;
subplot(3,1,2)
hold on
plot(theta_r,base_rad,'LineWidth',1)
plot(theta_r,Y_rad,'LineWidth',1)
hold off
legend('pre','post')
ylabel('Radial [mm]')
ax = gca;
ax.FontSize = 16;
subplot(3,1,3)
bar(theta_s, tension_m)
%ylim([800,1200])
legend('pre','post')
ylabel('Tension [N]')
xlabel('Rim Angle [rad]')
ax = gca;
ax.FontSize = 16;