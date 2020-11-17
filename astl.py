#!/usr/bin/env python
# coding: utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
from stl import mesh

"""
使用Graham扫描法计算凸包，从网上找到的代码，核心是求凸包
"""


def get_bottom_point(points):
    """
    返回points中纵坐标最小的点的索引，如果有多个纵坐标最小的点则返回其中横坐标最小的那个
    :param points:
    :return:
    """
    min_index = 0
    n = len(points)
    for i in range(0, n):
        if points[i][1] < points[min_index][1] or (
                points[i][1] == points[min_index][1] and points[i][0] < points[min_index][0]):
            min_index = i
    return min_index


def sort_polar_angle_cos(points, center_point):
    """
    按照与中心点的极角进行排序，使用的是余弦的方法
    :param points: 需要排序的点
    :param center_point: 中心点
    :return:
    """
    n = len(points)
    cos_value = []
    rank = []
    norm_list = []
    for i in range(0, n):
        point_ = points[i]
        point = [point_[0] - center_point[0], point_[1] - center_point[1]]
        rank.append(i)
        norm_value = math.sqrt(point[0] * point[0] + point[1] * point[1])
        norm_list.append(norm_value)
        if norm_value == 0:
            cos_value.append(1)
        else:
            cos_value.append(point[0] / norm_value)

    for i in range(0, n - 1):
        index = i + 1
        while index > 0:
            if cos_value[index] > cos_value[index - 1] or (
                    cos_value[index] == cos_value[index - 1] and norm_list[index] > norm_list[index - 1]):
                temp = cos_value[index]
                temp_rank = rank[index]
                temp_norm = norm_list[index]
                cos_value[index] = cos_value[index - 1]
                rank[index] = rank[index - 1]
                norm_list[index] = norm_list[index - 1]
                cos_value[index - 1] = temp
                rank[index - 1] = temp_rank
                norm_list[index - 1] = temp_norm
                index = index - 1
            else:
                break
    sorted_points = []
    for i in rank:
        sorted_points.append(points[i])

    return sorted_points


def vector_angle(vector):
    """
    返回一个向量与向量 [1, 0]之间的夹角， 这个夹角是指从[1, 0]沿逆时针方向旋转多少度能到达这个向量
    :param vector:
    :return:
    """
    norm_ = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])
    if norm_ == 0:
        return 0

    angle = math.acos(vector[0] / norm_)
    if vector[1] >= 0:
        return angle
    else:
        return 2 * math.pi - angle


def coss_multi(v1, v2):
    """
    计算两个向量的叉乘
    :param v1:
    :param v2:
    :return:
    """
    return v1[0] * v2[1] - v1[1] * v2[0]


def graham_scan(points):
    # print("Graham扫描法计算凸包")
    bottom_index = get_bottom_point(points)
    bottom_point = points.pop(bottom_index)
    sorted_points = sort_polar_angle_cos(points, bottom_point)

    m = len(sorted_points)
    if m < 2:
        print("点的数量过少，无法构成凸包")
        return

    stack = []
    stack.append(bottom_point)
    stack.append(sorted_points[0])
    stack.append(sorted_points[1])

    for i in range(2, m):
        length = len(stack)
        top = stack[length - 1]
        next_top = stack[length - 2]
        v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
        v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        while coss_multi(v1, v2) >= 0:
            stack.pop()
            length = len(stack)
            top = stack[length - 1]
            next_top = stack[length - 2]
            v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
            v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        stack.append(sorted_points[i])

    return stack


def line_from_point(p1, p2):  # AX+BY+C=0
    '''根据点求直线表达式'''
    X1, Y1 = p1
    X2, Y2 = p2

    A = Y2 - Y1
    B = X1 - X2
    C = X2 * Y1 - X1 * Y2
    return A, B, C


def dist_in_point_to_line(p, A, B, C):
    '''求点到直线的距离'''
    x, y = p
    return abs(A * x + B * y + C) / ((A ** 2 + B ** 2) ** 0.5)


stl = mesh.Mesh.from_file('L4_pedicle.stl')  # 加载模型文件

vectors = stl.vectors.reshape(-1, 3)  # 获取模型中的所有的点，每一个元素是由三个标量组成的向量，代表空间中一点

bottom_y = vectors.min(axis=0)[1]  # 获取最底下和最顶部的y值
top_y = vectors.max(axis=0)[1]

print(bottom_y, top_y)

vectors.min(axis=0), vectors.max(axis=0)

dys = np.linspace(bottom_y, top_y, int((top_y - bottom_y) / 0.05))  # 每隔0.5mm取一个


def filter_vec_by_y_thresh(vectors, y, thresh=0.25):
    result = []
    for vector in vectors:
        if abs(vector[1] - y) < thresh:
            result.append(vector)
    return np.array(result)


def filter_vec_by_degree(vectors, degree, thresh=5 * np.pi / 360):
    result = []
    for vec in vectors:
        sin_range = (np.sin(degree - thresh), np.sin(degree + thresh))
        cos_range = (np.cos(degree - thresh), np.cos(degree + thresh))

        bevel = np.linalg.norm((vec[0], vec[2]))  # 斜边长度

        sin = vec[0] / bevel
        cos = vec[2] / bevel

        if min(sin_range) < sin < max(sin_range):
            if min(cos_range) < cos < max(cos_range):
                result.append(vec)
    return np.array(result)


i = 0


def draw(points, result, center, min_dist_point):
    global i
    #     points = [[1.1, 3.6],
    #               [2.1, 5.4],
    #               [2.5, 1.8],
    #               [3.3, 3.98],
    #               [4.8, 6.2],
    #               [4.3, 4.1],
    #               [4.2, 2.4],
    #               [5.9, 3.5],
    #               [6.2, 5.3],
    #               [6.1, 2.56],
    #               [7.4, 3.7],
    #               [7.1, 4.3],
    #               [7, 4.1]]

    for point in points:
        plt.scatter(point[0], point[1], marker='.', c='y')

    length = len(result)
    for i in range(0, length - 1):
        plt.plot([result[i][0], result[i + 1][0]], [result[i][1], result[i + 1][1]], c='b')
    plt.plot([result[0][0], result[length - 1][0]], [result[0][1], result[length - 1][1]], c='b')

    center = np.array(result).mean(axis=0)

    plt.scatter(center[0], center[1])
    plt.scatter(min_dist_point[0], min_dist_point[1])

    plt.xlim((-30, 0))
    plt.ylim((-180, -150))
    plt.savefig("{}.png".format(i))
    i += 1
    plt.show()


# 每一个元素中包括了 最短距离（半轴长），求出最短距离的那个点，那个界面的中心点，那个界面的y值（高度值）
min_dist_and_point_and_center_and_y = []

axis_lengths = []
for cur_y in dys:  # 每隔0.5mm
    # 选择阈值小于0.25mm 的点
    results = filter_vec_by_y_thresh(vectors, cur_y)
    xz_points = results[:, [0, 2]]
    xz_points = xz_points.tolist()
    xz_set = []  # 去重，不然没法求凸包
    for i in xz_points:
        if i not in xz_set:
            xz_set.append(i)
    #     test1(xz_set)
    sort_ang_xz_points = graham_scan(xz_set)

    center = np.array(sort_ang_xz_points).mean(axis=0)

    dists = []

    for i in range(len(sort_ang_xz_points) - 1):
        A, B, C = line_from_point(sort_ang_xz_points[i], sort_ang_xz_points[i + 1])
        dist = dist_in_point_to_line(center, A, B, C)
        dists.append(dist)

    min_id = np.array(dists).argmin()
    min_dist_and_point_and_center_and_y.append((dists[min_id], sort_ang_xz_points[min_id], center, cur_y))
    draw(xz_points,sort_ang_xz_points,center,sort_ang_xz_points[min_id])
    print(dists[min_id], sort_ang_xz_points[min_id], center, cur_y)

print(min_dist_and_point_and_center_and_y)
