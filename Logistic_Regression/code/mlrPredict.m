function [label] = mlrPredict(W, X)
% blrObjFunction predicts the label of data given the data and parameter W
% of multi-class Logistic Regression
%
% Input:
% W: the matrix of weight of size (D + 1) x 10
% X: the data matrix of size N x D
%
% Output: 
% label: vector of size N x 1 representing the predicted label of
%        corresponding feature vector given in data matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_with_bias = [ones(size(X,1),1) X];

actfunc_matrix = exp(data_with_bias * W);
actfunc_summation = sum(actfunc_matrix,2);
actfunc_sum_repmatrix = repmat(actfunc_summation,1,size(actfunc_matrix,2));

y_n = actfunc_matrix./actfunc_sum_repmatrix;


[M,I] = max(y_n,[],2);

label = I;

end

