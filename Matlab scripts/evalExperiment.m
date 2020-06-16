function [delta_X,delta_tension] = evalExperiment(experiment)
% runExperiment.m:  Load baseline and displaced wheel data for a given
% experiment and evaluate against the model
% clear all
% close all

% Load the model--includes both symmetric and four-fold tension curves and
% assumed weighting factors mu1 and mu2
load('gainCurves.mat')
% Load experimental data.  This includes the baseline (pre-adjustment)
% data, the displaced data (post adjustment), and the adjustment vector, X,
% that was applied during the truing operation.
load(experiment)

% Generate a model that evaluates the variation of tension away from the
% mean tension.  i.e., Y_hat_tension = Phi*(X - mean(X)) + c0*mean(X). c0
% will be fit from experiments.
% choose which gain curves, either Phi_s (mean tension gain curves) or Phi
% (four-fold tension gain curves).

%Phi = Phi_s
% Note that sumsq error of tension is roughly 2x when using symmetric
% tension gain curves and the average tension is unchanged

numSpokes = size(X_adj,1);
% mean tension change for mean spoke adjustment (found from affineModel.m:
c = -472.95911;

Phi_lat = Phi(1:2*numSpokes,:);
Phi_rad = Phi(2*numSpokes+1:4*numSpokes,:);
Phi_ten = Phi(4*numSpokes+1:end,:);

Y_lat_pre = Y_pre(1:2*numSpokes);
Y_rad_pre = Y_pre(2*numSpokes+1:4*numSpokes);
Y_ten_pre = Y_pre(4*numSpokes+1:end);

Y_lat_post = Y_post(1:2*numSpokes);
Y_rad_post = Y_post(2*numSpokes+1:4*numSpokes);
Y_ten_post = Y_post(4*numSpokes+1:end);

delta_tension = mean(Y_ten_post - Y_ten_pre);
delta_X = mean(X_adj);
delta_T = c*delta_X;

Y_lat_hat = Phi_lat*X_adj + Y_lat_pre;
Y_rad_hat = Phi_rad*X_adj +Y_rad_pre;
Y_ten_hat = Phi_ten * (X_adj-delta_X) + delta_T+Y_ten_pre;

Y_lat_err = Y_lat_post - Y_lat_hat;
Y_rad_err = Y_rad_post - Y_rad_hat;
Y_ten_err = Y_ten_post - Y_ten_hat;

Y_hat = cat(1,Y_lat_hat,Y_rad_hat,Y_ten_hat);

plotExperiment(Y_hat, Y_post, Y_pre, 0)
fprintf('Model rms error:\n')
fprintf('Lateral: %1.3f \n',rms(Y_lat_err))
fprintf('Radial: %1.4f \n',rms(Y_rad_err))
fprintf('Tension: %1.0f \n',rms(Y_ten_err))
fprintf('Mean adjustment:  %1.3f \n', delta_X)
fprintf('Mean tension: %1.0f \n', mean(Y_ten_post))
fprintf('Mean tension change:  %1.0f \n', delta_tension)