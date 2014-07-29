function [error, error_grad] = mlrObjFunction(W, X, T)
% mlrObjFunction computes multi-class Logistic Regression error function 
% and its gradient.
%
% Input:
% W: the vector of size ((D + 1) * 10) x 1. Later on, it will reshape into
%    matrix of size D + 1) x 10
% X: the data matrix of size N x D
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
%
% Output: 
% error: the scalar value of error function of 2-class logistic regression
% error_grad: the vector of size ((D+1) * 10) x 1 representing the gradient 
%             of error function


W = reshape(W, size(X, 2) + 1, size(T, 2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_data_with_bias = [ones(size(X,1),1) X];

actfunc_matrix = exp(train_data_with_bias * W);
actfunc_summation = sum(actfunc_matrix,2);
actfunc_sum_repmatrix = repmat(actfunc_summation,1,size(actfunc_matrix,2));

y = actfunc_matrix./actfunc_sum_repmatrix;

%Computing the Error
error_matrix = T.*log(y);
error = -1*(sum(sum(error_matrix)));

%Computing the Gradiance
grad_matrix = ((y - T)'*train_data_with_bias)';

% Unroll gradients to single column vector
error_grad = grad_matrix(:);

end
