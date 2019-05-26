%% Initialization
clear ; close all; clc
addpath ("./functions")

# NN structure definition
input_layer_size = 8;
hidden_layer_size = 6;
output_layes_size = 1;
num_labels = 1;


#Loading data
train_src = "data/prepared_training.csv";
test_src = "data/prepared_test.csv";

train_data = csvread(train_src)(2:end,:);
test_data = csvread(test_src)(2:end,:);


X_test = test_data(:,4:end);
y_test = test_data(:,3);


X = train_data(:,4:end);
y = train_data(:,3);
m = size(X,1);


#Initial thetas values
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layes_size);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

# Training NN network using fminunc

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 200);

%  You should also try different values of lambda
lambda = 0.01;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

               
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Accuracy =================

save nn_params.mat nn_params;
save cost.mat cost;

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

pred = predict(Theta1, Theta2, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);