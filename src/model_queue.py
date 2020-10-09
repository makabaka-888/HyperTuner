# coding=utf-8
# Created on 2020-09-10 22:37
# Copyright © 2020 Alan. All rights reserved.
import random
import numpy as np
import math


class ModelQueue:
    """
    Model queue classes
    model_queue records [(model_path, init_lr, rank= -log(epoch) * acc, prob]
    the probability is acquired by softmax(rank_i)
    """
    def __init__(self, queue_size=3, log_lr_grid=[-2, -3, -4, -5]):
        # init_lr for fast converge, rank for evolutionary algorithm
        self.queue_size = queue_size
        self.log_lr_grid = log_lr_grid
        self.q = []

    def append_info(self, model_path, log_lr, train_params):
        """
        push new info into the queue
        if length surpasses the maximum queue size, pop the one with lowest probability
        :param model_path:
        :param log_lr:
        """
        rank = self.get_rank(train_params)
        self.q.append({"model_path": model_path, "log_lr": log_lr, "rank": rank, "prob": 0})
        self.q.sort(key=lambda x: x["rank"])

        # pop the lowest probability model
        if len(self.q) > self.queue_size:
            self.q.pop(0)
        self.update_prob()

    def k_crossover(self, k_value=2):
        """
        :param k_value: The number of values for model iteration
        :return: the model with the most frequency and the average log learning rate
        """
        prob_list = [self.q[i]["prob"] for i in range(len(self.q))]

        # model
        total_sum = 0

        # new_list should at the length of k_value
        new_list = []
        for i in range(k_value):
            rand_num = random.random()
            for key in range(len(prob_list)):
                total_sum += prob_list[key]
                if rand_num < total_sum:
                    new_list.append(key)
                    break

        # random.choices only in python3.6+
        # new_list = random.choices(self.q, weights=prob_list, k=k_value)

        all_log_lr = [self.q[i]["log_lr"] for i in new_list]
        avg_log_lr = sum(all_log_lr) / len(all_log_lr)

        # choose the model with the highest frequency in the choice list
        all_model_path = [self.q[i]["model_path"] for i in new_list]
        curr_model = max(set(all_model_path), key=all_model_path.count)

        return curr_model, avg_log_lr

    def update_prob(self):
        """
        softmax probability based on the ranking method
        probability = e^rank(i) / sum_k(e^rank(k))
        """
        tmp = []
        for i in range(len(self.q)):
            tmp.append(self.q[i]["rank"])

        tmp = np.array(tmp)

        # calculate the softmax probability based on ranking
        tmp = np.exp(tmp) / sum(np.exp(tmp))
        for i in range(len(self.q)):
            self.q[i]["prob"] = tmp[i]

    def get_rank(self, params, method="normal"):
        """
        customized ranking method
        """
        score, num_epcoch = params
        if method == "normal":
            return score
        elif method == "new":
            return -math.log(num_epcoch) * score
        return -1

