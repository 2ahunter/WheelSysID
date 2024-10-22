% Model the gain curve data using Fourier series and SVD
close all
load('gcData.mat')
load('gainCurves.mat')

%% Model the gain curves using a Fourier series

% gain curve data normalized to spoke 16
numSpokes = 32;
spoke=16;
theta = pi/32:pi/32:2*pi;
%% Randomly divide the data betwween model and test:
m=1;
n=1;
for col = 1:size(gc_lat,2)
    x = rand() - 0.5;
    if x<0
        gc_lat_model(:,m) = gc_lat(:,col);
        gc_rad_model(:,m) = gc_rad(:,col);
        m=m+1;
    else
        gc_lat_test(:,n) = gc_lat(:,col);
        gc_rad_test(:,n) = gc_rad(:,col);
        n = n+1;
    end
end
gc_lat_model = mean(gc_lat_model,2);
gc_rad_model = mean(gc_rad_model,2);
gc_rad_model = gc_rad_model -mean(gc_rad_model);
% Use the other half to test the model:
gc_lat_test = mean(gc_lat_test,2);
gc_rad_test = mean(gc_rad_test,2);
%%  fit to data
[A,a_coef] = fitData(gc_lat_model,spoke);
[B,b_coef] = fitData(gc_rad_model,spoke);

% test models
n = size(A,2);
y_l = gc_lat_model;
y_r = gc_rad_model;
y_lt = gc_lat_test;
y_rt = gc_rad_test - mean(gc_rad_test);
model_lr = zeros(n,1);
model_rr = zeros(n,1);
test_lr = model_lr;
test_rr = model_rr;
for i=1:n
    y_lhat = A(:,1:i)*a_coef(1:i);
    y_rhat = B(:,1:i)*b_coef(1:i);
    model_lr(i) = sum((y_l - y_lhat).^2,1);
    model_rr(i) = sum((y_r - y_rhat).^2,1);
    test_lr(i) = sum((y_lt - y_lhat).^2,1);
    test_rr(i) = sum((y_rt - y_rhat).^2,1);
end

%% Plot the residuals
figure(1)
subplot(2,1,1)
semilogy(model_lr,'k--')
hold on
semilogy(test_lr,'k-.')
title('Lateral Fit')
legend('Model','Test')
ylabel('Error [mm^2]')
ylim([10^-5 10^1])
ax = gca;
ax.FontSize=16;

subplot(2,1,2)
semilogy(model_rr,'k--')
hold on
semilogy(test_rr,'k-.')
hold off
title('Radial Fit')
legend('Model','Test')
ylabel('Error [mm^2]')
xlabel('Number of Fitting Coefficients')
ylim([10^-6 10^0])
ax = gca;
ax.FontSize=16;

figure(2) 
subplot(2,1,1)
plot(theta, y_lt, 'x')
hold on
plot(theta, A(:,1:13)*a_coef(1:13))
subplot(2,1,2)
plot(theta, y_rt, 'x')
hold on
plot(theta, B(:,1:26)*b_coef(1:26))

%% build IM with model rather than data
gc_lat_m = mean(gc_lat,2);
gc_rad_m = mean(gc_rad,2);

[L,l_coef] = fitData(gc_lat_m,spoke);
[R,r_coef] = fitData(gc_rad_m,spoke);

gcl_hat = L(:,1:13)*l_coef(1:13);
gcr_hat = R(:,1:27)*r_coef(1:27);

figure(3)
subplot(2,1,1)
plot(gc_lat_m,'x')
hold on
plot(gcl_hat,'--')
hold off
subplot(2,1,2)
plot(gc_rad_m,'x')
hold on
plot(gcr_hat,'--')
hold off

IM_lat_m = zeros(2*numSpokes,numSpokes);
IM_rad_m = IM_lat_m;
for spoke = 1:numSpokes
    s = rem(spoke,2);
    if s==1
        IM_lat_m(:,spoke) = shiftGC(gcl_hat, spoke);
        IM_rad_m(:,spoke) = shiftGC(gcr_hat, spoke);
    else
        IM_lat_m(:,spoke) = -shiftGC(gcl_hat, spoke);
        IM_rad_m(:,spoke) = shiftGC(gcr_hat, spoke);
    end
end

% verify gain curves:

figure(4)
subplot(2,1,1)
plot(IM_lat)
subplot(2,1,2)
plot(IM_rad)




%% Evaluate Influence Matrices

IM_lat = Phi(1:64,:);
IM_rad = Phi(65:128,:);
IM_ten = Phi(129:end,:);

[U_l,S_l,V_l] = svd(IM_lat);
[U_r,S_r,V_r] = svd(IM_rad);
[U_t,S_t,V_t] = svd(IM_ten);

Phi_lr = cat(1,IM_lat,IM_rad);
[U_lr,S_lr,V_lr] = svd(Phi_lr);

Phi_w = cat(1,IM_lat,IM_rad*mu1,IM_ten*mu2);
[U,S,V] = svd(Phi_w);

s_l = diag(S_l);
s_r = diag(S_r);
s_t = diag(S_t);
s_lr = diag(S_lr);
s = diag(S);


figure(5)
plot(s_l,'kx:')
hold on
plot(s_lr,'kd-.')
plot(s,'ko-')
hold off
title('Singular Values of Models')
legend('Lateral only','Lateral and Radial','Full Model')
ylabel('Magnitude [au]')
xlabel('Singular Value Order [int]')
ax = gca;
ax.FontSize=16;
