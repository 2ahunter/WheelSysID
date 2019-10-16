function y = plotExperiment(Y_hat, Y,base, target_ten)
% takes the baseline data, predicted data, and post-truing data as single
% column vectors.  target tension is a scalar.
y=1;
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

figure()
subplot(3,1,1)
hold on
plot(Y_lat_hat,'b')
plot(Y_lat,'bd')
hold off
subplot(3,1,2)
hold on
plot(Y_rad_hat,'r')
plot(Y_rad,'rd')
hold off
subplot(3,1,3)
hold on
bar(Y_ten_hat)
bar(Y_ten-target_ten)
hold off

figure()
subplot(3,1,1)
hold on
plot(Y_lat_hat- Y_lat,'rd-.')
hold off
subplot(3,1,2)
hold on
plot(Y_rad_hat- Y_rad,'rd-.')
hold off
subplot(3,1,3)
bar(Y_ten_hat - (Y_ten-target_ten))

spokes = 1:32;
tension_m = cat(2,base_ten+target_ten,Y_ten);
figure()
subplot(3,1,1)
hold on
plot(base_lat)
plot(Y_lat)
hold off
legend('pre-adjustment','post-adjustment')
title('Lateral')
subplot(3,1,2)
hold on
plot(base_rad)
plot(Y_rad)
hold off
title('Radial')
subplot(3,1,3)
hold on
bar(spokes, tension_m)
hold off
ylim([800,1200])
title('Tension')