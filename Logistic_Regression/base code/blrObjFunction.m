function [error, error_grad] = blrObjFunction(w, X, t)
% blrObjFunction computes 2-class Logistic Regression error function and
% its gradient.
%
% Input:
% w: the weight vector of size (D + 1) x 1 
% X: the data matrix of size N x D
% t: the label vector of size N x 1 where each entry can be either 0 or 1
%    representing the label of corresponding feature vector
%
% Output: 
% error: the scalar value of error function of 2-class logistic regression
% error_grad: the vector of size (D+1) x 1 representing the gradient of
%             error function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_data_with_bias = [ones(size(X,1),1) X];

y_n = sigmoid(train_data_with_bias * w);

%Computing the Error
error_matrix = ((t.*log(y_n))+((1-t).*log(1-y_n)));
error = -1*(sum(error_matrix));

%Computing the Gradiance
error_grad = ((y_n - t)'*train_data_with_bias)';
%error_grad = zeros(size(X, 2) + 1, 1); % dummy return


end
