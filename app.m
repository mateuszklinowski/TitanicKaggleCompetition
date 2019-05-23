train_src = "data/train.csv";
test_src = "data/test.csv";
train_data = csvread(train_src);
test_data = csvread(test_src);

size(test_data)
size(train_data)