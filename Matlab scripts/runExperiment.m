% runExperiment.m:  Load baseline and displaced wheel data for a given
% experiment and evaluate against the model

% Constants
% mean tension change for mean spoke adjustment (found from affineModel.m:
c = -472.95911;
numSpokes = 32;

% Load the model--includes both symmetric and four-fold tension curves and
% assumed weighting factors mu1 and mu2
load('gainCurves.mat')

% redefine tension weighting if you like:
mu2 = sqrt(5e-5);

% load pre-tuning data (the post data from previous experiment, usually
preNum = 10;
oldExperiment = strcat('exp',num2str(preNum),'.mat');
load(oldExperiment)
Y_pre = Y_post;

% baseline data:
Y_lat_pre = Y_pre(1:2*numSpokes);
Y_rad_pre = Y_pre(2*numSpokes+1:4*numSpokes);
Y_ten_pre = Y_pre(4*numSpokes+1:end);
Y_ten_pre_mean = mean(Y_ten_pre);

% separate state adjustment model into submatrices
Phi_lat = Phi(1:2*numSpokes,:);
Phi_rad = Phi(2*numSpokes+1:4*numSpokes,:);
Phi_ten = Phi(4*numSpokes+1:end,:);

% generate weighted vector measurements
Y_w = cat(1,Y_lat_pre,Y_rad_pre*mu1,(Y_ten_pre-Y_ten_pre_mean)*mu2);

%generate weighted matrix model Phi:
Phi_w = cat(1, Phi_lat,Phi_rad*mu1,Phi_ten*mu2);

%calculate weighted least square approximation to spoke vector:
d_hat = Phi_w\Y_w;
% invert sign to get truing vector and get ride of common mode 
d = -(d_hat - mean(d_hat));

% alternatively load a predefined set of spoke adjustments:
% load('X_test.mat')
% d = X - mean(X);


% Set target tension
% get dc component of tension from X
% delta_tension = mean(X)*c;
% or define a target tension :
target_tension = 1000;
delta_tension = target_tension - Y_ten_pre_mean;
d_cm = delta_tension/c;
% spoke adjustment vector
X_adj = d + d_cm;

Y_lat_hat = Phi_lat*d + Y_lat_pre;
Y_rad_hat = Phi_rad*d + Y_rad_pre;
Y_ten_hat = Y_ten_pre + Phi_ten*d + delta_tension;

figure()
subplot(3,1,1)
plot(Y_lat_hat)
subplot(3,1,2)
plot(Y_rad_hat)
subplot(3,1,3)
bar(Y_ten_hat)
ylim([800,1100])

% we won't use the predicted data returned by trueWheel, but we want to
% generate the truing recipe
Y_hat_2 = trueWheel(X_adj,Phi,Y_pre);

%% In this section, load the data from the experiment and store the results
% 
% load tension conversion table:
load('WFCompCal.mat');
D = WF_cal_18(:,1);
T = WF_cal_18(:,2);

% load lateral and radial displacements:
expNum = 11;
% filename for the post exerimental data
fn1 = strcat('valid_32_',num2str(expNum),'.mat');
load(fn1)
%load tension data:
fn2 = strcat('ten_valid_',num2str(expNum),'.mat');
load(fn2)
% convert tension displacements to tension units:
tension = spline(D,T,ten_d);

% % data:
Y_lat_post = data_LR(1,:)';
Y_rad_post = data_LR(2,:)';
Y_ten_post = tension;
% put into single vector for exp file
Y_post = cat(1,Y_lat_post,Y_rad_post,Y_ten_post);


%% save experiment here
% newExperiment = strcat('exp',num2str(expNum),'.mat')
% save(newExperiment,'target_tension','X_adj','Y_post','Y_pre')
