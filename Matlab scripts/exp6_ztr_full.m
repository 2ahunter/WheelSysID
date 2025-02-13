%% ZTR Full Experiment
% Measure each spoke individually
% baseline and spoke 1-2 calibration
% calibration image in data\ztr_final\calibration\
% cal_b12.JPG
% gauge centers: c1: 1765, 1408; c2: 2041, 3356
% cal_3_14.JPG
% gauge centers: c1: 1763, 1394; c2: 2037, 3342
% cal_15_27.JPG
% gauge centers: c1: 1766, 1429; c2: 2040, 3378
% cal_28_32.JPG
% gauge centers: c1: 1767, 1428; c2: 2041, 3378
% baseline data taken before spoke 1, before spoke 5, before spoke 29,
% after spoke32
% gauge centers 

%% load workspace
close all
clear all
load('ztr_final.mat')
load('tensionztrall.mat')
numSpokes = 32;
j = 1:2*numSpokes;
theta = j*2*pi/64;
% figure()
% plot(matrix_tension_raw)

%% Convert raw tensiometer displacements into tension data
% calibration curve for tensiometer for 1.8mm spokes.  Displacement in mm
% is the first column tension in NM is second column
load('WFCompCal.mat');
d = WF_cal_18(:,1);
T = WF_cal_18(:,2);
% figure()
% plot(WF_cal_18(:,1),WF_cal_18(:,2),'x-')
maxCol = size(matrix_tension_raw,2);
matrix_tension = zeros(numSpokes,maxCol);
for col = 1:maxCol
    matrix_tension(:,col) = spline(d,T,matrix_tension_raw(:,col));
end
% figure()
% plot(matrix_tension(:,1:3),'x-.')
% baseline data is the first three columns:
base_tension = mean(matrix_tension(:,1:3),2);
% gain curves after each spoke is loosened one turn minus minus mean
% baseline
gc_tension = matrix_tension(:,4:end) - base_tension;
% figure
% bar(gc_tension(:,1))

% create four types of spokes:
% non-drive leading spokes
NDL = zeros(numSpokes,numSpokes/4);
% drive side leading
DL = NDL;
% non-drive trailing spoke
NDT = NDL;
% drive-side trailing spokes
DT = NDL;

for i = 1:8
    j = (i-1)*4;
    NDL(:,i) = shiftTC(gc_tension(:,j+1),32+16 - (i-1)*4);
    DL(:,i) = shiftTC(gc_tension(:,j+2),31+16 - (i-1)*4);
    NDT(:,i) = shiftTC(gc_tension(:,j+3),30+16 - (i-1)*4);
    DT(:,i) = shiftTC(gc_tension(:,j+4),29+16 - (i-1)*4);
end

%% tension gain curves
% four configurations:  non-drive leading, drive leading, non-drive
% trailing and drive trailing
NDL_mean = mean(NDL,2);
DL_mean = mean(DL,2);
NDT_mean = mean(NDT,2);
DT_mean = mean(DT,2);

T_mean = mean(cat(2,NDL_mean,DL_mean,NDT_mean,DT_mean),2);
% bias_T = -5;
% T_mean = T_mean-bias_T;
% scale_ten = 2;
% T_mean = T_mean * scale_ten;
% figure()
% subplot(4,1,1)
% bar(NDL_mean)
% subplot(4,1,2)
% bar(DL_mean)
% subplot(4,1,3)
% bar(NDT_mean)
% subplot(4,1,4)
% bar(DT_mean)

% there are two distinct orientations of spoke tension gain curves in a
% three cross wheel.  One that is more symmetric and one that is more
% asymmetric.  Call them sym_ten and asym_ten.  non-drive leading and
% drive-side trailing are asymmetric and mirrored with respect to each
% other.  DL and NDT are (more) symmetric and also mirror images

asym_ten = (NDL_mean + shiftTC(flip(DT_mean),1))/2;
sym_ten = (DL_mean + shiftTC(flip(NDT_mean),1))/2;

c = -473;  % measured average tension change for all spokes loosened by 1 turn
offset_asym = c/numSpokes - mean(NDL_mean);
offset_sym = c/numSpokes - mean(DL_mean);
% adjust for bias
asym_ten_off = (NDL_mean + shiftTC(flip(DT_mean),1))/2 + offset_asym;
sym_ten_off = (DL_mean + shiftTC(flip(NDT_mean),1))/2 + offset_sym;

theta_s = pi/32:pi/16:2*pi;
figure(1)
subplot(4,1,1)
b = bar(theta_s,asym_ten);
b(1).EdgeColor = 'k';
b(1).FaceColor = [0.5,0.5,0.5];
title('Tension Gain Curves')
ylabel('Tension [N]')
ax = gca; % current axes
ax.FontSize = 16;
ylim([-250,150])
subplot(4,1,2)
b=bar(theta_s,sym_ten);
b(1).EdgeColor = 'k';
b(1).FaceColor = [0.5,0.5,0.5];
ylabel('Tension [N]')
ax = gca; % current axes
ax.FontSize = 16;
ylim([-250,150])
subplot(4,1,3)
b = bar(theta_s,shiftTC(flip(sym_ten),1));
b(1).EdgeColor = 'k';
b(1).FaceColor = [0.5,0.5,0.5];
ylabel('Tension [N]')
ax = gca; % current axes
ax.FontSize = 16;
ylim([-250,150])
subplot(4,1,4)
b= bar(theta_s,shiftTC(flip(asym_ten),1));
b(1).EdgeColor = 'k';
b(1).FaceColor = [0.5,0.5,0.5];
xlabel('Rim Angle [rad]')
ylabel('Tension [N]')
ax = gca; % current axes
ax.FontSize = 16;
ylim([-250,150])

% Generate influence matrix gain curves for each spoke:
IM_ten = zeros(numSpokes,numSpokes);
IM_ten_off = zeros(numSpokes,numSpokes);
IM_mean_ten = IM_ten;
for spoke = 1:numSpokes
    s = rem(spoke,4);
    % symmetric approximation to get rid of tension discretization
    IM_mean_ten(:,spoke) = shiftTC(T_mean,15+spoke);
    % gain curves using all distinct patterns (leads to errors due to
    % discretization!
    if s==1
        IM_ten(:,spoke) = shiftTC(asym_ten, 15+spoke);
        IM_ten_off(:,spoke) = shiftTC(asym_ten_off, 15+spoke);
    elseif s==2
        IM_ten(:,spoke) = shiftTC(sym_ten, 15+spoke);
        IM_ten_off(:,spoke) = shiftTC(sym_ten_off, 15+spoke);
    elseif s==3
        IM_ten(:,spoke) = shiftTC(flip(sym_ten), 15+spoke+1);
        IM_ten_off(:,spoke) = shiftTC(flip(sym_ten_off), 15+spoke+1);
    else
        IM_ten(:,spoke) = shiftTC(flip(asym_ten), 15+spoke +1);
        IM_ten_off(:,spoke) = shiftTC(flip(asym_ten_off), 15+spoke +1);
    end
end

% Conclusions: IM_mean_ten doesn't fit as well as IM_ten.  Adding offset to
% try to bring mean tension to match experimental data doesn't work.  To
% see this multiply IM_ten_off a vector of ones.  The answer should be
% uniformly -473, but instead we get the repeated pattern of  
% -561.6751, -435.2975, -435.2975, -561.6751

%% Evaluate baseline data

base_lat = cat(2, matrix_base(:,1),matrix_base(:,3),matrix_base(:,5),matrix_base(:,7));
base_rad = cat(2, matrix_base(:,2),matrix_base(:,4),matrix_base(:,6),matrix_base(:,8));
base_ten = matrix_tension_raw(:,1:3);
% test repeatability of baseline measurements
% figure(2)
% subplot(3,1,1)
% plot(base_lat,'x-')
% title('Baseline Measurements Taken During Wheel Characterization')
% subplot(3,1,2)
% plot(base_rad,'x-')
% subplot(3,1,3)
% bar(base_ten)

% use mean baseline data
base_lat_mean = mean(base_lat,2);
base_rad_mean = mean(base_rad,2);
base_ten_mean = mean(matrix_tension_raw(:,1:3),2);

% what is variation between mean and each bseline?
delta_base_lat = base_lat - base_lat_mean;
delta_base_rad = base_rad - base_rad_mean;
delta_base_ten = base_ten - base_ten_mean;

% plot repeatability data
% figure(3)
% 
% subplot(3,1,1)
% plot(delta_base_lat,'x')
% title('Repeatbility of Baseline Measurements')
% subplot(3,1,2)
% plot(delta_base_rad,'x')
% subplot(3,1,3)
% plot(delta_base_ten,'x-.')


%% Evaluate rim displacement data

% figure()
% subplot(2,1,1)
% plot(theta, matrix_lat,'x-')
% subplot(2,1,2)
% plot(theta, matrix_rad,'x-')
    
%% Calculate raw influence matrix curves

im_lat_raw = matrix_lat - base_lat_mean;
im_rad_raw = matrix_rad - base_rad_mean;
theta_r = pi/32:pi/32:2*pi;

% figure()
% subplot(2,1,1)
% plot(theta, im_lat_raw,'-')
% subplot(2,1,2)
% plot(theta, im_rad_raw,'-')

%plot gain curves shifted to spoke 16
gc_lat = zeros(2*numSpokes,numSpokes);
gc_rad = gc_lat;

figure()
subplot(2,1,1)
hold on
for spoke = 1:numSpokes
    data = im_lat_raw(:,spoke);
    if rem(spoke,2) ==1
        gc_lat(:,spoke) = gCurve(data, spoke);
    else
        gc_lat(:,spoke) = -gCurve(data, spoke);
    end
    gc = shiftGC(gc_lat(:,spoke),16);
    plot(theta_r,gc)
end
hold off
title('Lateral Gain Curve Measurements')
legend('Lateral')
ylabel('Displacement [mm]')
ax = gca;
ax.FontSize=16;

subplot(2,1,2)
hold on
for spoke = 1:numSpokes
    data = im_rad_raw(:,spoke);
    gc_rad(:,spoke) = gCurve(data, spoke);
    gc = shiftGC(gc_rad(:,spoke),16);
    plot(theta_r,gc)
end
hold off
title('Radial Gain Curve Measurements')
legend('Radial')
ylabel('Displacement [mm]')
xlabel('Rim Angle [rad]')
ax = gca;
ax.FontSize=16;
    
% generate mean gain curves
gc_lat_mean = mean(gc_lat,2);
gc_rad_mean = mean(gc_rad,2);
% eliminate bias from rad
bias_rad = mean(gc_rad_mean);
gc_rad_mean = gc_rad_mean - bias_rad;

figure()
plot(theta_r,shiftGC(gc_lat_mean,16),'kx-.','LineWidth',1)
hold on
plot(theta_r,shiftGC(gc_rad_mean,16),'ko-.','LineWidth',1)
hold off
title('Mean Gain Curves')
legend('Lateral','Radial')
ylabel('Displacement [mm]')
xlabel('Rim Angle [rad]')
ax = gca;
ax.FontSize=16;

% generate IM displacement gain curves

IM_lat = zeros(2*numSpokes,numSpokes);
IM_rad = IM_lat;
for spoke = 1:numSpokes
    s = rem(spoke,2);
    if s==1
        IM_lat(:,spoke) = shiftGC(gc_lat_mean, spoke);
        IM_rad(:,spoke) = shiftGC(gc_rad_mean, spoke);
    else
        IM_lat(:,spoke) = -shiftGC(gc_lat_mean, spoke);
        IM_rad(:,spoke) = shiftGC(gc_rad_mean, spoke);
    end
end

% verify gain curves:

% figure()
% subplot(2,1,1)
% plot(IM_lat)
% subplot(2,1,2)
% plot(IM_rad)

%% Wheel Truing Simulation 
% radial to lateral exchange rate
mu1 = sqrt(0.25);
% tension to lateral exchange rate
mu2 = sqrt(1.0e-5);
% weighted Influence Matrices
Phi_w = cat(1,IM_lat,mu1*IM_rad,mu2*IM_ten);
% weighted IM using symmetric tension gain curves
Phi_sw = cat(1, IM_lat, mu1*IM_rad,mu2*IM_mean_ten);
% unweighted IM
Phi = cat(1,IM_lat,IM_rad,IM_ten);
% unweighted IM with symmetric tension curves
Phi_s = cat(1,IM_lat,IM_rad,IM_mean_ten);


% X is the adjustment vector
X = zeros(numSpokes,1);
% generate random spoke turn vector
pd = makedist('Normal','mu',0,'sigma',0.67);
X = random(pd,[numSpokes,1]);
X = round(X*4);
X = X/4;
% To test with the experimental version for the next section uncomment:
% load('X_test.mat')
% Y is the measurement vector
% new random lateral error vector
pd = makedist('Normal','mu',0,'sigma',0.03);
YLat = IM_lat*X + random(pd,[2*numSpokes,1]);
% new random radial error vector
pd = makedist('Normal','mu',0,'sigma',0.01);
YRad = IM_rad*X + random(pd,[2*numSpokes,1]);
% new random tension error vector
pd = makedist('Normal','mu',0,'sigma',20);
% use the symmetric tension curves by default
Y_ten = IM_mean_ten*X + random(pd,[numSpokes,1]);
% Y_ten = IM_mean*X + random(pd,[numSpokes,1]);
Y_w = cat(1,YLat,mu1*YRad,mu2*Y_ten);
%  Again symmetric tension curves are default.  
X_hat = Phi_sw\Y_w;
% X_hat = Phi_w\Y_w;

%% 

% plot experiment
figure()
subplot(3,1,1)
plot(theta_r,YLat,'LineWidth',1)
title('Predicted Profiles for Random De-Truing')
ylabel('Lateral [mm]')
ax = gca;
ax.FontSize=16;
subplot(3,1,2)
plot(theta_r,YRad,'LineWidth',1)
ylabel('Radial [mm]')
ax = gca;
ax.FontSize=16;
subplot(3,1,3)
bar(theta_s,Y_ten)
ylabel('Tension [N]')
xlabel('Angle [rad]')
ax = gca;
ax.FontSize=16;
% compare prediction versus actual
figure()
hold on
stem(X,'k','LineWidth',1)
stem(X_hat,'Color', [0.5 0.5 0.5],'LineWidth',1)
hold off
title('Simulated Truing Solution')
legend('actual','predicted')
ylabel('Adjustment [revs]')
xlabel('Spoke Number')
ax = gca;
ax.FontSize=16;

% Evaluate final result
figure()
subplot(4,1,1)
plot(theta_r,IM_lat*X_hat-YLat)
title('Lateral Error')
ylabel('Lateral [mm]')
ax = gca;
ax.FontSize=16;
subplot(4,1,2)
plot(theta_r,IM_rad*X_hat-YRad)
title('Radial Error')
ylabel('Radial [mm]')
ax = gca;
ax.FontSize=16;
subplot(4,1,3)
bar(theta_s,IM_ten*X_hat-Y_ten)
title('Tension Error')
ylabel('Tension [N]')
ax = gca;
ax.FontSize=16;
subplot(4,1,4)
stem(theta_s, X_hat - X)
title('Spoke Adjustment Error')
ylabel('Error [revs]')
xlabel('Angle [rad]')
ax = gca;
ax.FontSize=16;


%% Estimate baseline correction
target_tension = 800;
mean_tension = mean(base_tension);
delta_ten = base_tension-target_tension;

Y_w = cat(1,base_lat(:,4),mu1*base_rad(:,4),mu2*(delta_ten));
X_hat = Phi_w\Y_w;
% just to verify that ML isn't doing something sneaky:
%X_hat2 = pinv(Phi)*Y;
X_adj = X_hat;


%% Validate model experimentally (exp 1)
load('X_test.mat')
load('baseline_5.mat')
% gauge centers:  1664 1393 1938 3316
baseline_lat = baseline_5(1,:)';
baseline_rad = baseline_5(2,:)';
base_x1 = cat(1,baseline_lat,baseline_rad,base_tension);
% data taken after adjust first four spokes
load('valid_4.mat')
v4_lat = valid_4(1,:)';
v4_rad = valid_4(2,:)';
% data taken after adjust first 16 spokes
load('valid_16.mat')
v16_lat = valid_16(1,:)';
v16_rad = valid_16(2,:)';
% data taken after adjusting all spokes
load('valid_32.mat')
v32_lat = valid_32(1,:)';
v32_rad = valid_32(2,:)';
% tension data taken only at the end
load('tension_valid.mat')
% process tension data
v32_ten = spline(d,T,tension_validation_d);
% save experiment data
X_adj = X;
Y_pre = base_x1;
Yx1 = cat(1,v32_lat,v32_rad,v32_ten);
Y_post = Yx1;
save('exp1.mat','Y_pre','Y_post','X_adj','target_tension')

Y_x1 = valid_32';

% Baseline data
% figure()
% subplot(2,1,1)
% plot(baseline_lat,'x-')
% subplot(2,1,2)
% plot(baseline_rad,'x-');
% generate Y_est at each step sequentially from first spoke adjustment to
% the last.  Validate the spoke adjustment with lateral measurement

X_temp = zeros(numSpokes,1);
Y_at_index = X_temp;

% containers for data:
Y_lat_hat = zeros(2*numSpokes, numSpokes);
Y_rad_hat = zeros(2*numSpokes, numSpokes);
Y_ten_hat = zeros(numSpokes, numSpokes);

% generate the incremental curves to validate each adjustment
for spoke = 1:numSpokes
    index = 2*spoke -1;
    X_adj = cat(1,X(1:spoke),X_temp(spoke+1:end));
    Y_est = Phi*X_adj;
    Y_lat_hat(:,spoke) = Y_est(1:64)+baseline_lat;
    Y_rad_hat(:,spoke) = Y_est(65:128)+baseline_rad;
    Y_ten_hat(:,spoke) = Y_est(129:end)+base_tension;
    Y_at_index(spoke) = Y_lat_hat(index);
    if X_adj(spoke)>0
        fprintf('loosen %d, %1.2f turns, %1.3f, \n',spoke,X_adj(spoke), Y_at_index(spoke))
    elseif X_adj(spoke)<0
        fprintf('tighten %d,%1.2f turns, %1.3f, \n',spoke,X_adj(spoke), Y_at_index(spoke))
    else
        fprintf('no adjustment of %d \n',spoke)
    end
end


% Y_est_x1 = trueWheel(X,Phi,base_x1);
% split data into components
% Y_lat_hat = Y_est_x1(1:64);
% Y_rad_hat = Y_est_x1(65:128);
% Y_ten_hat = Y_est_x1(129:end);
figure()
subplot(3,1,1)
hold on
% plot(Y_lat_hat(:,4),'b-')
% plot(v4_lat,'bx')
% plot(Y_lat_hat(:,16),'g-')
% plot(v16_lat,'go')
plot(theta_r,Y_lat_hat(:,32),'k-.','LineWidth',1)
plot(theta_r,v32_lat,'kx','LineWidth',1)
hold off
ylabel('Lateral [mm]')
legend('Predict','Measure')
title('Prediction and Measurement of Random De-truing')
ax = gca;
ax.FontSize = 16;
subplot(3,1,2)
hold on
% plot(Y_rad_hat(:,4),'b-')
% plot(v4_rad,'bx')
% plot(Y_rad_hat(:,16),'g-')
% plot(v16_rad,'go')
plot(theta_r,Y_rad_hat(:,32),'k-.','LineWidth',1)
plot(theta_r,v32_rad,'kx','LineWidth',1)
hold off
ax = gca;
ax.FontSize = 16;
ylabel('Radial [mm]')
legend('Predict','Measure')
subplot(3,1,3)
hold on
data = cat(2,Y_ten_hat(:,32),v32_ten);
% bar(theta_s,Y_ten_hat(:,32))
b = bar(theta_s,data,1,'hist');
b(1).EdgeColor = 'k';
b(1).FaceColor = [0.5,0.5,0.5];
b(2).EdgeColor = 'k';
b(2).FaceColor = 'w';
hold off
ax = gca;
ax.FontSize = 16;
ylabel('Tension [N]')
xlabel('Rim Angle [rad]')
legend('Predict','Measure')

figure()
subplot(3,1,1)
plot(theta_r,Y_lat_hat(:,32)- v32_lat,'kx:','LineWidth',1)
ylabel('Lateral [mm]')
title('Model Error')
ax = gca;
ax.FontSize = 16;
subplot(3,1,2)
plot(theta_r,Y_rad_hat(:,32)- v32_rad,'kx:','LineWidth',1)
ylabel('Radial [mm]')
ax = gca;
ax.FontSize = 16;
subplot(3,1,3)
b = bar(theta_s,Y_ten_hat(:,32) - v32_ten,0.8);
b(1).EdgeColor = 'k';
b(1).FaceColor = [0.5 0.5 0.5];
ylabel('Tension [N]')
xlabel('Rim Angle [rad]')
ax = gca;
ax.FontSize = 16;
max(Y_lat_hat(:,32)- v32_lat) - min(Y_lat_hat(:,32)- v32_lat);
max(abs(Y_rad_hat(:,32)- v32_rad));
%% True the wheel from the previous experiment
% measurement matrix with weighting:
Y_w = cat(1,v32_lat,mu1*v32_rad,mu2*(v32_ten-target_tension));
Y = cat(1,v32_lat,v32_rad,v32_ten-target_tension);
X_hat = Phi_w\Y_w;
X_adj = -X_hat;
X_w_hat = Phi_sw\Y_w;

%compare against disturbance--they should be different
% figure()
% hold on
% stem(X_adj)
% stem(X_hat)
% stem(X_w_hat)
% hold off
% legend('actual','asymmetric','symmetric')
% % evaluate prediction
% Y_lat_hat = v32_lat-IM_lat*X_hat;
% Y_rad_hat = v32_rad-IM_rad*X_hat;
% Y_ten_hat = v32_ten-IM_ten*X_hat;
% 
% figure()
% subplot(3,1,1)
% plot(Y_lat_hat)
% subplot(3,1,2)
% plot(Y_rad_hat)
% subplot(3,1,3)
% bar(Y_ten_hat-target_tension)
% 
% Y_lat_hat = v32_lat-IM_lat*X_w_hat;
% Y_rad_hat = v32_rad-IM_rad*X_w_hat;
% Y_ten_hat = v32_ten-IM_mean_ten*X_w_hat;
% 
% figure()
% subplot(3,1,1)
% plot(Y_lat_hat)
% subplot(3,1,2)
% plot(Y_rad_hat)
% subplot(3,1,3)
% bar(Y_ten_hat-target_tension)


%% Generate lateral targets for truing operation
% Use solution from previous section, changing sign to account for
% reversing the operation

% redefine the baseline to the previous section result:
baseline_lat = v32_lat;
baseline_rad = v32_rad;
baseline_ten = v32_ten - target_tension;

baseline_x2 = cat(1,baseline_lat,baseline_rad,baseline_ten);
Y_hat_x2 = trueWheel(-X_hat,Phi_s,baseline_x2);
Y_hat=Y_hat_x2(:,end);
% split data into components
Y_lat_hat = Y_hat(1:64);
Y_rad_hat = Y_hat(65:128);
Y_ten_hat = Y_hat(129:end);

% figure()
% subplot(3,1,1)
% plot(Y_lat_hat)
% subplot(3,1,2)
% plot(Y_rad_hat)
% subplot(3,1,3)
% bar(Y_ten_hat)

%% Evaluate result
% load raw data after all adjustments
load('ten_valid_2.mat')
load('valid_32_2.mat')
v32_ten = spline(d,T,ten_d);
v32_lat = valid_32_2(1,:)';
v32_rad = valid_32_2(2,:)';
Y_x2 = cat(1,v32_lat,v32_rad,v32_ten);
% pre-state:
Y_pre = Yx1;
Y_post = Y_x2;
save('exp2.mat','Y_pre','Y_post','X_adj','target_tension')

plotExperiment(Y_hat,Y_post,Y_pre,target_tension)


%% Iterate on new baseline
% measurement matrix with weighting:
% take out the mean in radial data--we don't care about the absolute value
% only the variation
Y_w = cat(1,v32_lat,mu1*(v32_rad-mean(v32_rad)),mu2*(v32_ten-target_tension));
X_hat = Phi_w\Y_w;

% look at solution
% figure()
% stem(-X_hat)

% redefine the baseline to the previous section result:
baseline_lat = v32_lat;
baseline_rad = v32_rad;
baseline_ten = v32_ten - target_tension;
baseline_x3 = cat(1,baseline_lat,baseline_rad,baseline_ten);
% run truing program
Y_hat_x3 = trueWheel(-X_hat,Phi_s,baseline_x3);
Y_hat = Y_hat_x3(:,end);
% split data into components
Y_lat_hat = Y_hat(1:64);
Y_rad_hat = Y_hat(65:128);
Y_ten_hat = Y_hat(129:end);

%% Evaluate results for second truing (experiment 3)

load('ten_valid_3.mat');
% convert to tension units
v32_ten = spline(d,T,ten_d);
load('valid_32_3.mat')
v32_lat = valid_32_3(1,:)';
v32_rad = valid_32_3(2,:)';

Y_x3 = cat(1,v32_lat,v32_rad,v32_ten);
X_adj = -X_hat;
Y_pre = Y_x2;
Y_post = Y_x3;
save('exp3.mat','Y_pre','Y_post','X_adj','target_tension')
plotExperiment(Y_hat,Y_post,Y_pre,target_tension)


%% Change target tension and retrue the wheel (experiment 4)
target_tension = 1000;
Y_w = cat(1,v32_lat,mu1*(v32_rad),(mu2)*(v32_ten-target_tension));
X_hat = Phi_w\Y_w;
X_adj = -X_hat;
X_hat_x4 = X_hat;
Y_est = Phi_s*(-X_hat);
% figure()
% subplot(4,1,1)
% stem(-X_hat)
% subplot(4,1,2)
% plot(Y_est(1:64)+v32_lat)
% subplot(4,1,3)
% plot(Y_est(65:128)+v32_rad)
% subplot(4,1,4)
% bar(Y_est(129:end)+v32_ten)

% redefine the baseline to the previous section result:
baseline_x4 = Y_x3;
baseline_x4(129:end) = baseline_x4(129:end)-target_tension;
% run truing program
Y_hat_x4 = trueWheel(-X_hat,Phi,baseline_x4);
Y_hat = Y_hat_x4(:,end);

%% Evaluate results for re-tension truing validation test 4 (new target tension)

load('ten_valid_4.mat');
v32_ten = ten_valid_4t;
load('valid_32_4.mat')
v32_lat = valid_32_4(1,:)';
v32_rad = valid_32_4(2,:)';
Y_x4 = cat(1,v32_lat,v32_rad,v32_ten);
Y_pre = Y_x3;
Y_post = Y_x4;
save('exp4.mat','Y_pre','Y_post','X_adj','target_tension')
plotExperiment(Y_hat,Y_post,Y_pre,target_tension)

%% Iterate truing solution based on new baseline data (experiment 5)
% This experiment is because the tension gain curves have significant error
% particularly when changing tension of an already true wheel.  Running the
% wheel twice fixes much of the issue, evaluate whether adding both spoke
% adjustment vectors yields a better solution for just changing tension

Y_w = cat(1,v32_lat,mu1*(v32_rad),(mu2)*(v32_ten-target_tension));
% Y_w = cat(1,v32_lat,(v32_rad-mean(v32_rad)))
% Phi_w = cat(1,IM_lat,IM_rad)
X_hat = Phi_w\Y_w;
X_hat_x5 = X_hat;
Y_est = Phi*(-X_hat);
% figure()
% subplot(4,1,1)
% stem(-X_hat)
% subplot(4,1,2)
% plot(Y_est(1:64)+v32_lat)
% subplot(4,1,3)
% plot(Y_est(65:128)+v32_rad)
% subplot(4,1,4)
% bar(Y_est(129:end)+v32_ten)

% redefine the baseline to the previous section result:
baseline_lat = v32_lat;
baseline_rad = v32_rad;
baseline_ten = v32_ten - target_tension;
baseline_x5 = cat(1,baseline_lat,baseline_rad,baseline_ten);
% run truing program
Y_hat_x5 = trueWheel(-X_hat,Phi,baseline_x5);
Y_hat = Y_hat_x5(:,end);


%% Evaluate results for re-tension truing validation test 5 (2nd iteration of new target tension)

load('ten_valid_5.mat');
ten_valid_5t = spline(d,T,ten_d);
v32_ten = ten_valid_5t;
load('valid_32_5.mat')
v32_lat = valid_32_5(1,:)';
v32_rad = valid_32_5(2,:)';
Y_x5 = cat(1,v32_lat,v32_rad,v32_ten);
X_adj = -X_hat;
Y_pre = Y_x4;
Y_post = Y_x5;
save('exp5.mat','Y_pre','Y_post','X_adj','target_tension')
plotExperiment(Y_hat,Y_post,Y_pre,target_tension)

X_hat_net = X_hat_x4 + X_hat_x5;
% figure()
% subplot(3,1,1)
% stem(X_hat_x4)
% subplot(3,1,2)
% stem(X_hat_x5)
% subplot(3,1,3)
% stem(X_hat_net)
%% try with symmetrical tension gain curves--experiment 6
Phi_s = cat(1,IM_lat,IM_rad,IM_mean_ten);
% weighted symmetric tension gain curves
Phi_sw = cat(1, IM_lat, mu1*IM_rad,mu2*IM_mean_ten);

% Define the baseline to the previous section result:
baseline=Y_x5;
baseline(129:end) = baseline(129:end) - target_tension;
Y_w = cat(1,baseline(1:64),mu1*(baseline(65:128)-mean(baseline(65:128))),mu2*baseline(129:end));

% Solve for spoke adjustments
X_hat = Phi_sw\Y_w;
X_hat_x6 = X_hat;
Y_est = Phi_s*(-X_hat);
% figure()
% subplot(4,1,1)
% stem(-X_hat)
% subplot(4,1,2)
% plot(Y_est(1:64)+baseline(1:64))
% subplot(4,1,3)
% plot(Y_est(65:128)+baseline(65:128))
% subplot(4,1,4)
% bar(Y_est(129:end)+baseline(129:end)+target_tension)

% todo: see if subtracting radial baseline significantly changes solution.
% It should contribute to an offset.  Or should we predict the offset
% change due to rim contraction and include a radial target?

Y_hat_x6 = trueWheel(-X_hat,Phi_s,baseline);
Y_hat = Y_hat_x6(:,end);
load('valid_32_6.mat');
load('ten_valid_6.mat');
ten_valid_6t = spline(d,T,ten_d);
v32_ten = ten_valid_6t;
v32_lat = valid_32_6(1,:)';
v32_rad = valid_32_6(2,:)';
Y_x6 = cat(1,v32_lat,v32_rad,v32_ten);
X_adj = -X_hat;
Y_pre = Y_x5;
Y_post = Y_x6;
save('exp6.mat','Y_pre','Y_post','X_adj','target_tension')
plotExperiment(Y_hat,Y_post,Y_pre,target_tension)


%% Plot Truing Algorithm example for experiment 2
Y_lat_hat = Y_hat_x2(1:64,:)';
Y_lat_b=Y_x1(1:64);
Y_at_index = zeros(numSpokes, 1);
for spoke = 1:numSpokes
    index = 2*spoke - 1;
    Y_at_index(spoke) = Y_lat_hat(spoke, index);
end

figure()
hold on
plot(theta_r, Y_lat_b,'LineWidth',3) 
hold off
xlim([0,2.03*pi])
ylabel('Lateral [mm]')
xlabel('Rim Angle [rad]')
title('Truing Algorithm: Step 0')
ax = gca;
ax.FontSize = 16;

figure()
hold on
plot(theta_r, Y_lat_b) 
plot(theta_r, Y_lat_hat(1,:),'LineWidth',3)
plot(theta_s(1),Y_at_index(1),'kd','MarkerSize',10,'LineWidth',2)
hold off
xlim([0,2.03*pi])
ylabel('Lateral [mm]')
xlabel('Rim Angle [rad]')
title('Truing Algorithm: Step 1')
ax = gca;
ax.FontSize = 16;

figure()
hold on
plot(theta_r, Y_lat_b) 
plot(theta_r, Y_lat_hat(1,:),'LineWidth',0.5)
plot(theta_r, Y_lat_hat(2,:),'LineWidth',3)
plot(theta_s(2),Y_at_index(2),'kd','MarkerSize',10,'LineWidth',2)
hold off
xlim([0,2.03*pi])
ylabel('Lateral [mm]')
xlabel('Rim Angle [rad]')
title('Truing Algorithm: Step 2')
ax = gca;
ax.FontSize = 16;

figure()
hold on
plot(theta_r, Y_lat_b) 
plot(theta_r, Y_lat_hat(1:2,:),'LineWidth',0.5)
plot(theta_r, Y_lat_hat(3,:),'LineWidth',3)
plot(theta_s(3),Y_at_index(3),'kd','MarkerSize',10,'LineWidth',2)
hold off
xlim([0,2.03*pi])
ylabel('Lateral [mm]')
xlabel('Rim Angle [rad]')
title('Truing Algorithm: Step 3')
ax = gca;
ax.FontSize = 16;

figure()
hold on
plot(theta_r, Y_lat_b) 
plot(theta_r,Y_lat_hat(1:31,:))
plot(theta_r,Y_lat_hat(32,:),'LineWidth',3)
plot(theta_s(32),Y_at_index(32), 'kd','MarkerSize',10,'LineWidth',2)
hold off
xlim([0,2.03*pi])
ylabel('Lateral [mm]')
xlabel('Rim Angle [rad]')
title('Truing Algorithm: Step 32')
ax = gca;
ax.FontSize = 16;

figure()
hold on
plot(theta_r, Y_lat_b,'-.','Color',[0.5 0.5 0.5])
plot(theta_r,Y_lat_hat(1:31,:),'Color',[0.5 0.5 0.5]) 
plot(theta_r,Y_lat_hat(32,:),'k','LineWidth',3)
plot(theta_s(1:31),Y_at_index(1:31), 'kd','MarkerSize',10)
plot(theta_s(32),Y_at_index(32), 'kd','MarkerSize',10,'LineWidth',3)
hold off
xlim([0,2.03*pi])
ylabel('Lateral [mm]')
xlabel('Rim Angle [rad]')
title('Predicted Lateral Curves and Adjustment Points for Truing Algorithm')
ax = gca;
ax.FontSize = 16;



%% Simualate tension change with both tension models:

Y_lat_sim = zeros(64,1);
Y_rad_sim = zeros(64,1);
Y_ten_sim = -200 * ones(32,1);
Y_sim = cat(1,Y_lat_sim,Y_rad_sim,Y_ten_sim);

%four fold tension gain curves:
X_hat = Phi\Y_sim;
% symmetric gain curves:
X_hat_s = Phi_s\Y_sim;

figure()
stem(theta_s,X_hat,'LineWidth',2)
hold on
stem(theta_s,X_hat_s,'LineWidth',2)
hold off
legend('full model', 'average model')
xlabel('Rim angle')
ylabel('Spoke Adjustment [revs]')
title('Simulation of Tension Models')
ylim([0.4,0.9])
xlim([0,2*pi])
ax = gca;
ax.FontSize = 16;

%% Simulate one spoke turn
X = ones(32,1);
Y = zeros(160,1);
Y(129:end) = 1050;
scale_ten = 1.525;
Phi_test = cat(1,IM_lat,IM_rad,IM_mean_ten*scale_ten);
Y_hat = Y + Phi_test*X;

figure()
subplot(3,1,1)
plot(theta_r,Y_hat(1:64))
subplot(3,1,2)
plot(theta_r,Y_hat(65:128))
subplot(3,1,3)
bar(theta_s,Y_hat(129:end));

%% Regularized least squares--is a symmetric tension model as good as regularized LS?
% radial to lateral exchange rate
mu1 = sqrt(0.5);
% tension to lateral exchange rate
mu2 = sqrt(1.0e-5);
% Regularization exchange rate:
mu3 = sqrt(0.0005);
% Regularization phi:
Phi_rw = cat(1, IM_lat, mu1*IM_rad,mu3*eye(32));
% Regularization phi:
Phi_r = cat(1, IM_lat,IM_rad,eye(32));

% X is the adjustment vector
X = zeros(numSpokes,1);
% generate random spoke turn vector
pd = makedist('Normal','mu',0,'sigma',0.67);
X = random(pd,[numSpokes,1]);
X = round(X*4);
X = X/4;
% To test with the experimental version for the next section uncomment:
load('X_test.mat')
% Y is the measurement vector
% new random lateral error vector
pd = makedist('Normal','mu',0,'sigma',0.03);
YLat = IM_lat*X + random(pd,[2*numSpokes,1]);
% new random radial error vector
pd = makedist('Normal','mu',0,'sigma',0.01);
YRad = IM_rad*X + random(pd,[2*numSpokes,1]);
% new random tension error vector
pd = makedist('Normal','mu',0,'sigma',20);
Y_ten = IM_mean_ten*X + random(pd,[numSpokes,1]);
Y_w = cat(1,YLat,mu1*YRad,mu2*Y_ten);
Y_rw = cat(1,YLat,mu1*YRad,zeros(32,1));
% Predicted spoke turns using symmetric tension gain curve
X_hat_s = Phi_sw\Y_w;
% Predicted spoke turns using regularization
X_hat_r = Phi_rw\Y_rw;

%% plot experiment
% figure()
% subplot(3,1,1)
% plot(theta_r,YLat,'LineWidth',1)
% title('Predicted Profiles for Random De-Truing')
% ylabel('Lateral [mm]')
% ax = gca;
% ax.FontSize=16;
% subplot(3,1,2)
% plot(theta_r,YRad,'LineWidth',1)
% ylabel('Radial [mm]')
% ax = gca;
% ax.FontSize=16;
% subplot(3,1,3)
% bar(theta_s,Y_ten)
% ylabel('Tension [N]')
% xlabel('Angle [rad]')
% ax = gca;
% ax.FontSize=16;
% compare prediction versus actual
figure()
hold on
stem(X,'ko','LineWidth',1)
stem(X_hat_r,'kd','LineWidth',1)
stem(X_hat_s,'kx','LineWidth',1)

hold off
ylim([-1.5 1.5])
title('Simulated Truing Solutions')
legend('actual','regularized','full model','Location','southeast')
ylabel('Adjustment Prediction [revs]')
xlabel('Spoke Number')
ax = gca;
ax.FontSize=16;

% Evaluate final result
figure()
stem(X_hat_r - X,'kd')
hold on
stem(X_hat_s - X,'kx')
hold off

title('Simulated Adjustment Error')
legend('regularized','full model','Location','southeast')
ylabel('Error [revs]')
xlabel('Spoke Number')
ax = gca;
ax.FontSize=16;


%% Experiment 7:  Adjust by one rotation looser using feedback
baseline = Y_x6; % current wheel state
target_tension = mean(baseline(129:end));
baseline(129:end) = baseline(129:end) - target_tension;

X = ones(32,1); % one rotation
X_adj = X;
Y_hat_x7 = trueWheel(X,Phi_s,baseline);
Y_hat = Y_hat_x7(:,end);
load('ten_valid_7.mat')
ten_valid_7t = spline(d,T,ten_d);
load('valid_32_7.mat')
v32_ten = ten_valid_7t;
v32_lat = valid_32_7(1,:)';
v32_rad = valid_32_7(2,:)';
Y_x7 = cat(1,v32_lat,v32_rad,v32_ten);
Y_pre = Y_x6;
Y_post = Y_x7;
save('exp7.mat','Y_pre','Y_post','X_adj','target_tension')
plotExperiment(Y_hat,Y_post,Y_pre,target_tension);






