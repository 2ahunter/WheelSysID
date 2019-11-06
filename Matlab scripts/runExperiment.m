% runExperiment.m:  Load baseline and displaced wheel data for a given
% experiment and evaluate against the model
%clear all
close all
% Constants
% mean tension change for mean spoke adjustment (found from affineModel.m:
c = -472.95911;
numSpokes = 32;
% Set target tension
target_tension = 1000;

% load tension conversion table:
load('WFCompCal.mat');
d = WF_cal_18(:,1);
T = WF_cal_18(:,2);

% Load the model--includes both symmetric and four-fold tension curves and
% assumed weighting factors mu1 and mu2
load('gainCurves.mat')

% Load basline data. 
load('valid_32_7.mat')
load('ten_valid_7.mat')

% convert tension displacements to tension units:
ten_valid_7t = spline(d,T,ten_valid_7);

% baseline data:
Y_lat_pre = valid_32_7(1,:)';
Y_rad_pre = valid_32_7(2,:);
Y_ten_pre = ten_valid_7t - mean(ten_valid_7t);
% put in a single vector for later use:
baseline = cat(1,Y_lat_pre,Y_rad_pre,Y_ten_pre);

% separate state adjustment model into submatrices
Phi_lat = Phi(1:2*numSpokes,:);
Phi_rad = Phi(2*numSpokes+1:4*numSpokes,:);
Phi_ten = Phi(4*numSpokes+1:end,:);

% generate weighted vector measurements
Y_w = cat(1,Y_lat_pre,Y_rad_pre*mu1,Y_ten_pre*mu2);

%generate weighted matrix model Phi:
Phi_w = cat(1, Phi_lat,Phi_rad*mu1,Phi_ten*mu2);

%calculate weighted least square approximation to spoke vector:
d_hat = Phi_w\Y_w;
% invert sign to get truing vector
d = -d_hat;

delta_tension = mean(Y_ten_post - Y_ten_pre);
delta_d = mean(d);
delta_T = c*delta_d;

Y_lat_hat = Phi_lat*d + Y_lat_pre;
Y_rad_hat = Phi_rad*d + Y_rad_pre;
Y_ten_hat = Phi_ten * (d-delta_d) + delta_T+Y_ten_pre;

%we won't use the predicted data returned by trueWheel, but we want to
%generate the truing recipe
Y_hat_2 = trueWheel(d,Phi,baseline);
