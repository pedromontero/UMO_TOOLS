#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io


def tsearch_arbitrary(p, t, x, y):

    p = p['p']
    p = p.transpose()
    t = t['t']
    t = t[0:3, :]

    x = x['x']
    y = y['y']
    s = x.shape
    print(s)


    pass


def main():
    path = './demo/tsearch_arbitrary/'
    p = scipy.io.loadmat(path + 'p.mat')
    t = scipy.io.loadmat(path + 't.mat')
    x = scipy.io.loadmat(path + 'x.mat')
    y = scipy.io.loadmat(path + 'y.mat')
    tsearch_arbitrary(p, t, x, y)




if __name__ == '__main__':
    main()



