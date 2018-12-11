#!/usr/bin/python
# # coding:utf-8


if __name__ == "__main__":
    import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    kk = 1