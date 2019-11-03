% runExperiment.m:  Load baseline and displaced wheel data for a given
% experiment and evaluate against the model

% Load the model--includes both symmetric and four-fold tension curves and
% assumed weighting factors mu1 and mu2
load('gainCurves.mat')
% Load experimental data.  This includes the baseline (pre-adjustment)
% data, the displaced data (post adjustment), and the adjustment vector, X,
% that was applied during the truing operation.
load('exp1.mat')

% Generate a model that evaluates the variation of tension away from the
% mean tension.  i.e., Y_hat_tension = Phi*(X - mean(X)) + c0*mean(X). c0
% will be fit from experiments.