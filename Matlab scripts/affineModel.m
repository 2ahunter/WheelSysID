% Affine model:  Trying to fit a model of the form y_hat = Phi*X +
% c0*mean(X).  Find c0:

numExp = 7;
X = zeros(numExp,1);
DT = zeros(numExp,1);
for expNum = 1:numExp
    exp = strcat('exp',num2str(expNum),'.mat');
    [X(expNum),DT(expNum)] = runExperiment(exp);
end
    
close all



%% 
x = (-1:0.1:1)';

A = X;
c = A\DT;

A_tilde = x;
DT_hat = A_tilde*c;



figure()
plot(x,DT_hat)
hold on
plot(X(:,1),DT,'x')
hold off

fprintf('Best fit constant = %1.5f \n', c(1))

%% 




    
