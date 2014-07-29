function [w] = blrNewtonRaphsonLearn(initial_w, X, t, n_iter)
%blrNewtonRaphsonLearn learns the weight vector of 2-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_w: vector of size (D+1) x 1 where D is the number of features in
%            feature vector
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector   
% t: vector of size N x 1 where each entry is either 0 or 1 representing
%    the true label of corresponding feature vector.
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% w: vector of size (D+1) x 1, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_data_with_bias = [ones(size(X,1),1) X];

w_old = initial_w;

for i = 1 : n_iter
     y_n = sigmoid(train_data_with_bias * w_old);
     r_n = y_n.*(1-y_n);
     r_nn = repmat(r_n',size(train_data_with_bias,2),1);
     H = ((train_data_with_bias'.*r_nn)*train_data_with_bias);
     w_old = w_old - mldivide(H , (train_data_with_bias'*(y_n-t))) ;
end

w = w_old;

%w = mldivide((train_data_with_bias'*train_data_with_bias),(train_data_with_bias'*t)); % see whether the iterative method should be implemented

end
