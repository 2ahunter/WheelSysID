function y = plotExperiment(Y_hat, Y, base, delta_ten)
% Input:  prediction, measurement, baseline, change in tension.  
% returns 1: 
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

f = figure();
figNum = f.Number;
subplot(3,1,1)
hold on
plot(theta_r,Y_lat_hat,'k-.','LineWidth',1)
plot(theta_r,Y_lat,'kx','LineWidth',1)
hold off
ylabel('Lateral [mm]')
legend('Predict','Measure')
title('Prediction and Measurement of Truing Operation')
ax = gca;
ax.FontSize = 16;

subplot(3,1,2)
hold on
plot(theta_r,Y_rad_hat,'k-.','LineWidth',1)
plot(theta_r,Y_rad,'kx','LineWidth',1)
hold off
ylabel('Radial [mm]')
legend('Predict','Measure','Location','southeast')
ax = gca;
ax.FontSize = 16;
subplot(3,1,3)
data = cat(2,Y_ten_hat+delta_ten,Y_ten);
b = bar(theta_s,data,1);
b(1).EdgeColor = 'k';
b(1).FaceColor = [0.5,0.5,0.5];
b(2).EdgeColor = 'k';
b(2).FaceColor = 'w';
ylabel('Tension [N]')
xlabel('Rim Angle [rad]')
legend('Predict','Measure')
ax = gca;
ax.FontSize = 16;

% plot model error
figure()
subplot(3,1,1)
plot(theta_r,Y_lat_hat- Y_lat,'kx-.','LineWidth',1)
ylabel('Lateral [mm]')
title('Model Error')
ax = gca;
ax.FontSize = 16;

subplot(3,1,2)
plot(theta_r,Y_rad_hat- Y_rad,'kx-.','LineWidth',1)
ylabel('Radial [mm]')
ax = gca;
ax.FontSize = 16;

subplot(3,1,3)
b= bar(theta_s,Y_ten_hat - (Y_ten-delta_ten),0.8);
b(1).EdgeColor = 'k';
b(1).FaceColor = [0.5,0.5,0.5];
ylabel('Tension [N]')
xlabel('Rim Angle [rad]')
ax = gca;
ax.FontSize = 16;

tension_m = cat(2,base_ten,Y_ten);
figure()
subplot(3,1,1)
hold on
plot(theta_r,base_lat,'kx-.','LineWidth',1)
plot(theta_r,Y_lat,'d-.','LineWidth',1,'Color',[0.5 0.5 0.5])
hold off
legend('pre','post')
ylabel('Lateral [mm]')
title('Measurements Before and After Truing Operation')
ax = gca;
ax.FontSize = 16;
subplot(3,1,2)
hold on
plot(theta_r,base_rad,'kx-.','LineWidth',1)
plot(theta_r,Y_rad,'d-.','LineWidth',1,'Color',[0.5 0.5 0.5])
hold off
legend('pre','post')
ylabel('Radial [mm]')
ax = gca;
ax.FontSize = 16;
subplot(3,1,3)
b = bar(theta_s, tension_m,1);
b(1).EdgeColor = 'k';
b(1).FaceColor = [0.5,0.5,0.5];
b(2).EdgeColor = 'k';
b(2).FaceColor = 'w';

%ylim([600,1200])
legend('pre','post')
ylabel('Tension [N]')
xlabel('Rim Angle [rad]')
ax = gca;
ax.FontSize = 16;

% print performance specs (formatted for tabular in latex:
fprintf('Figure %d performance\n',figNum)
fprintf('Lateral [mm] & $%1.3f\\pm%1.3f$ &',mean(base_lat),std(base_lat))
fprintf('$%1.3f\\pm %1.3f$ \\\\ \\hline \n',mean(Y_lat),std(Y_lat))
fprintf('Radial [mm] &$%1.3f\\pm%1.3f$& ',mean(base_rad),std(base_rad))
fprintf('$%1.3f\\pm%1.3f$ \\\\ \\hline \n',mean(Y_rad),std(Y_rad))
fprintf('Tension [N] &$%1.0f\\pm%1.0f$& ',mean(base_ten),std(base_ten))
fprintf('$%1.0f\\pm%1.0f$ \\\\ \\hline \n',mean(Y_ten),std(Y_ten))

%%1.2f
