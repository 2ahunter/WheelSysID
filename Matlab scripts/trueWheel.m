function Y = trueWheel(X,Phi,baseline)
% X is spoke adjustment vector, Phi is the Influence Matrix, baseline is
% the lateral, radial, and delta tension in a single vector
numSpokes = 32;
% split the baseline data into components:
baseline_lat = baseline(1:64);
baseline_rad = baseline(65:128);
baseline_ten = baseline(129:end);

% split the Phi matrix into submatrices:
Phi_lat = Phi(1:2*numSpokes,:);
Phi_rad = Phi(2*numSpokes+1:4*numSpokes,:);
Phi_ten = Phi(4*numSpokes+1:end,:);

% containers for data:
X_temp = zeros(numSpokes,1);
Y_at_index = X_temp;
Y_lat_hat = zeros(2*numSpokes, numSpokes);
Y_rad_hat = zeros(2*numSpokes, numSpokes);
Y_ten_hat = zeros(numSpokes, numSpokes);

% generate the incremental curves to validate each adjustment
for spoke = 1:numSpokes
    index = 2*spoke -1;
    X_adj = cat(1,X(1:spoke),X_temp(spoke+1:end));
    Y_lat_hat(:,spoke) = Phi_lat*X_adj+baseline_lat;
    Y_rad_hat(:,spoke) = Phi_rad*X_adj +baseline_rad;
    Y_ten_hat(:,spoke) = Phi_ten*X_adj+baseline_ten;
    Y_at_index(spoke) = Y_lat_hat(index,spoke);
    if X_adj(spoke)>0
        fprintf('loosen %d, %1.2f turns, %1.3f, \n',spoke,X_adj(spoke), Y_at_index(spoke))
    elseif X_adj(spoke)<0
        fprintf('tighten %d,%1.2f turns, %1.3f, \n',spoke,X_adj(spoke), Y_at_index(spoke))
    else
        fprintf('no adjustment of %d \n',spoke)
    end
end

Y = cat(1,Y_lat_hat,Y_rad_hat,Y_ten_hat);
end