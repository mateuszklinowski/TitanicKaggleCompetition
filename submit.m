%% Initialization
clear ; close all; clc
addpath ("./functions")

# NN structure definition
input_layer_size = 12;
hidden_layer_size = 1;
output_layes_size = 1;
num_labels = 1;

#Loading data
data_src = "data/submit_test.csv";

data = csvread(data_src)(2:end,:);

X = data(:,3:end);

m = size(X,1);

% Load cost
load cost.mat;
% Load nn_params
load nn_params.mat;

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X, 0.7);

submition = [data(:,2) pred];

csvwrite("submition.csv", submition);