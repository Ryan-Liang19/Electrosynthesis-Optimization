import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LinearRegression
from pyDOE2 import lhs
import cma


# sigma = 0.2357170341856802, 0.007664835601141715, 0.002617568712357526    
if __name__ == '__main__':
    df = pd.read_excel(io='experimental result.xlsx', header=0)
    data = np.array(df)
    x = data[:, 1:6]
    y = data[:, 8]
    gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-2, normalize_y=True, n_restarts_optimizer=100, random_state=1537)
    gpr.fit(x, y)
    y1 = data[:, 6]
    y2 = data[:, 7]
    gpr1 = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-2, normalize_y=True, n_restarts_optimizer=100, random_state=1537)
    gpr2 = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-2, normalize_y=True, n_restarts_optimizer=100, random_state=1537)
    gpr1.fit(x, y1)
    gpr2.fit(x, y2)
    t_budget = 900

    for seed in range(0, 50):
        np.random.seed(4567 + seed)

        def experiment(x):
            res1 = (gpr.predict(x) + np.random.normal(0, 0.2357170341856802, x.shape[0])).reshape(-1, 1)
            res2 = (gpr1.predict(x) + np.random.normal(0, 0.007664835601141715, x.shape[0])).reshape(-1, 1)
            res3 = (gpr2.predict(x) + np.random.normal(0, 0.002617568712357526, x.shape[0])).reshape(-1, 1)
            res = np.concatenate([res1, res2, res3], axis=1)
            return res

        def black_box_function(x):
            M = 200
            y = x[:, 0].copy()
            temp = x[:, 1]
            y -= M * (temp < 0.2) * (0.2 - temp)
            temp = x[:, 2]
            y -= M * (temp < 0.85) * (0.85 - temp)
            return y

        def cost_function(x, x_before):
            y = 220.7 / x[2] + 930.1
            if x[1] != x_before[1]:
                y += 697.2
            if x[3] != x_before[3]:
                y += 242.4
            return y / 60

        def cost_f_true(x, x_before):
            y = 220.7 / x[:, 2] + 930.1
            y += (x[:, 1] != x_before[1]) * 697.2
            y += (x[:, 3] != x_before[3]) * 242.4
            return y / 60
        # coefficient: 185.3, 902.8, 540.0, 16.6; 220.7, 930.1, 697.2, 242.4

        xx1 = np.arange(1.5, 3.01, 0.02)
        xx2 = np.array([0.3, 0.5, 0.8])
        xx3 = np.arange(0.1, 1.01, 0.01)
        xx4 = np.array([0.15, 0.3, 0.5])
        xx5 = np.concatenate([np.arange(0.25, 0.3333, 0.005), np.arange(0.3333, 0.5, 0.01), np.arange(0.5, 1, 0.015), np.arange(1, 3.03, 0.03)])
        xx1, xx2, xx3, xx4, xx5 = np.meshgrid(xx1, xx2, xx3, xx4, xx5)
        xx = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1), xx3.reshape(-1, 1), xx4.reshape(-1, 1), xx5.reshape(-1, 1)], axis=1)

        # method 1: BO alone
        # index_num = np.random.randint(0, xx.shape[0], 10)
        # x_obs = xx[index_num, :]
        # res_set = experiment(x_obs)
        # y_obs = black_box_function(res_set)
        # x_bef = np.array([0, 0, 0, 0, 0])
        # iter = 0
        # t = 0
        # t_set = np.empty(0)
        # for x_now in x_obs:
        #     t += cost_function(x_now, x_bef) + 20
        #     x_bef = x_now
        #     t_set = np.concatenate([t_set, [t]])
        #     iter += 1

        # gpr_bayers = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-2, normalize_y=True, n_restarts_optimizer=20, random_state=1537+seed)
        # while iter < 30:
        #     gpr_bayers.fit(x_obs, y_obs)
        #     mean, std = gpr_bayers.predict(xx, return_std=True)
        #     a = mean - max(y_obs)
        #     z = a / std
        #     ei = a * norm.cdf(z) + std * norm.pdf(z)
        #     index = np.argmax(ei)
        #     x_new = xx[index, :]
        #     res = experiment(x_new.reshape(1, -1))
        #     y_new = black_box_function(res)
        #     t += cost_function(x_new, x_bef) + 20
        #     x_obs = np.concatenate([x_obs, x_new.reshape(1, -1)])
        #     y_obs = np.concatenate([y_obs, y_new])
        #     res_set = np.concatenate([res_set, res.reshape(1, -1)])
        #     x_bef = x_new
        #     t_set = np.concatenate([t_set, [t]])
        #     print('Iteration {0} finished!'.format(iter))
        #     iter += 1

        # method 2: the proposed BO
        # penalization = 10
        # index_num = np.random.randint(0, xx.shape[0], 10)
        # x_obs = xx[index_num, :]
        # res_set = experiment(x_obs[:-1, :])
        # y_obs = black_box_function(res_set)
        # x_bef = np.array([0, 0, 0, 0, 0])
        # iter = 0
        # t = 0
        # t_set = np.empty(0)
        # for x_now in x_obs:
        #     t += max(cost_function(x_now, x_bef), 20)
        #     x_bef = x_now
        #     t_set = np.concatenate([t_set, [t]])
        #     iter += 1
        # gpr_bayers1 = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-2, normalize_y=True,
        #                                        n_restarts_optimizer=20, random_state=1537 + seed)
        # gpr_bayers2 = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-2, normalize_y=True,
        #                                        n_restarts_optimizer=20, random_state=1537 + seed)
        # true_set = experiment(x_obs[-1, :].reshape(1, -1))
        # fake_set = np.zeros((1, 3))
        # fake_set[0, 0] = true_set[0, 0]
        # gpr_bayers1.fit(x_obs[:-1, :], res_set[:, 1])
        # fake_set[0, 1] = gpr_bayers1.predict(x_obs[-1, :].reshape(1, -1))
        # gpr_bayers2.fit(x_obs[:-1, :], res_set[:, 2])
        # fake_set[0, 2] = gpr_bayers2.predict(x_obs[-1, :].reshape(1, -1))
        # y_fake = black_box_function(fake_set)
        # res_set = np.concatenate([res_set, fake_set])
        # y_obs = np.concatenate([y_obs, y_fake])
        #
        # gpr_bayers = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-2, normalize_y=True,
        #                                       n_restarts_optimizer=20, random_state=1537 + seed)
        # while iter < 40:
        #     gpr_bayers.fit(x_obs, y_obs)
        #     pen = max(penalization * (1 - t_set[-1] / t_budget), 0)
        #     mean, std = gpr_bayers.predict(xx, return_std=True)
        #     a = mean - max(y_obs)
        #     z = a / std
        #     ei = a * norm.cdf(z) + std * norm.pdf(z)
        #     t_est = cost_f_true(xx, x_obs[-1, :])
        #     pen *= np.std(ei) / np.mean(t_est)
        #     ei = a * norm.cdf(z) + std * norm.pdf(z) - pen * t_est
        #     index = np.argmax(ei)
        #     while ei[index] < 0:
        #         pen *= 0.7
        #         ei = a * norm.cdf(z) + std * norm.pdf(z) - pen * t_est
        #         index = np.argmax(ei)
        #
        #     x_new = xx[index, :]
        #     res_set[-1, :] = true_set
        #     y_obs[-1] = black_box_function(true_set)
        #
        #     true_set = experiment(x_new.reshape(1, -1))
        #     fake_set[0, 0] = true_set[0, 0]
        #     gpr_bayers1.fit(x_obs, res_set[:, 1])
        #     fake_set[0, 1] = gpr_bayers1.predict(x_new.reshape(1, -1))
        #     gpr_bayers2.fit(x_obs, res_set[:, 2])
        #     fake_set[0, 2] = gpr_bayers2.predict(x_new.reshape(1, -1))
        #     x_obs = np.concatenate([x_obs, x_new.reshape(1, -1)])
        #     res_set = np.concatenate([res_set, fake_set])
        #     y_fake = black_box_function(fake_set)
        #     y_obs = np.concatenate([y_obs, y_fake])
        #
        #     t += max(cost_function(x_new, x_bef), 20)
        #     x_bef = x_new
        #     t_set = np.concatenate([t_set, [t]])
        #     print('Iteration {0} finished!'.format(iter))
        #     iter += 1

        # method 3: the proposed BO (single objective)
        # penalization = 10
        # index_num = np.random.randint(0, xx.shape[0], 10)
        # x_obs = xx[index_num, :]
        # res_set = experiment(x_obs)
        # y_obs = res_set[:, 0]
        # x_bef = np.array([0, 0, 0, 0, 0])
        # iter = 0
        # t = 0
        # t_set = np.empty(0)
        # for x_now in x_obs:
        #     t += cost_function(x_now, x_bef)
        #     x_bef = x_now
        #     t_set = np.concatenate([t_set, [t]])
        #     iter += 1
        #
        # gpr_bayers = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-2, normalize_y=True,
        #                                       n_restarts_optimizer=20, random_state=1537 + seed)
        # while iter < 40:
        #     gpr_bayers.fit(x_obs, y_obs)
        #     pen = max(penalization * (1 - t_set[-1] / t_budget), 0)
        #     mean, std = gpr_bayers.predict(xx, return_std=True)
        #     a = mean - max(y_obs)
        #     z = a / std
        #     ei = a * norm.cdf(z) + std * norm.pdf(z)
        #     t_est = cost_f_true(xx, x_obs[-1, :])
        #     pen *= np.std(ei) / np.mean(t_est)
        #     ei = a * norm.cdf(z) + std * norm.pdf(z) - pen * t_est
        #     index = np.argmax(ei)
        #     while ei[index] < 0:
        #         pen *= 0.7
        #         ei = a * norm.cdf(z) + std * norm.pdf(z) - pen * t_est
        #         index = np.argmax(ei)
        #
        #     x_new = xx[index, :]
        #     y_new = experiment(x_new.reshape(1, -1))
        #     x_obs = np.concatenate([x_obs, x_new.reshape(1, -1)])
        #     res_set = np.concatenate([res_set, y_new])
        #     y_obs = np.concatenate([y_obs, y_new[:, 0]])
        #
        #     t += cost_function(x_new, x_bef)
        #     x_bef = x_new
        #     t_set = np.concatenate([t_set, [t]])
        #     print('Iteration {0} finished!'.format(iter))
        #     iter += 1

        # method 4: OVAT method
        # x_obs = np.array([[1.5, 0.3, 0.5, 0.15, 1]])
        # res_set = experiment(x_obs)
        # y_obs = black_box_function(res_set)
        # x_bef = np.array([0, 0, 0, 0, 0])
        # iter = 0
        # t = 0
        # t_set = np.empty(0)
        # for x_now in x_obs:
        #     t += cost_function(x_now, x_bef) + 20
        #     x_bef = x_now
        #     t_set = np.concatenate([t_set, [t]])
        #     iter += 1
        # order = np.arange(0, 5, 1)
        # np.random.shuffle(order)
        #
        # x_var = [[2.0, 2.5, 3], [0.5, 0.8], [0.1, 0.3, 0.7, 1], [0.3, 0.5], [0.25, 0.333, 0.5, 2, 3]]
        # x_new = x_obs[-1].copy()
        # for i in order:
        #     for j in x_var[i]:
        #         x_new[i] = j
        #         res = experiment(x_new.reshape(1, -1))
        #         y_new = black_box_function(res)
        #         t += cost_function(x_new, x_bef) + 20
        #         x_obs = np.concatenate([x_obs, x_new.reshape(1, -1)])
        #         y_obs = np.concatenate([y_obs, y_new])
        #         res_set = np.concatenate([res_set, res.reshape(1, -1)])
        #         x_bef = x_new
        #         t_set = np.concatenate([t_set, [t]])
        #         print('Iteration {0} finished!'.format(iter))
        #         iter += 1
        #     index = np.argmax(y_obs)
        #     x_new = x_obs[index]

        # method 5: HTS (random search)
        # index_num = np.random.randint(0, xx.shape[0], 6)
        # x_busy = xx[index_num, :]
        # x_bef = np.zeros((6, 5))
        # x_obs = np.empty((1, 5))
        # res_set = np.empty((1, 3))
        # y_obs = np.empty(0)
        # t_left = np.zeros(6)
        # iter = 0
        # t = 0
        # t_set = np.empty(0)
        # for i in range(0, 6):
        #     t_left[i] = cost_function(x_busy[i], x_bef[i]) + 20
        #
        # while iter < 150:
        #     index = np.argmin(t_left)
        #     t_temp = t_left[index]
        #     t_left -= t_temp
        #     t += t_temp
        #     t_set = np.concatenate([t_set, [t]])
        #     x_obs = np.concatenate([x_obs, x_busy[index].reshape(1, -1)])
        #     res = experiment(x_busy[index].reshape(1, -1))
        #     res_set = np.concatenate([res_set, res])
        #     y_new = black_box_function(res)
        #     y_obs = np.concatenate([y_obs, y_new])
        #
        #     x_bef[index] = x_busy[index]
        #     index_num = np.random.randint(0, xx.shape[0], 1)
        #     x_busy[index] = xx[index_num, :]
        #     t_left[index] = cost_function(x_busy[index], x_bef[index]) + 20
        #     print('Iteration {0} finished!'.format(iter))
        #     iter += 1
        # x_obs = np.delete(x_obs, 0, axis=0)
        # res_set = np.delete(res_set, 0, axis=0)

        # method 6: HTS (Particle swarm optimization)
        # index_num = np.random.randint(0, xx.shape[0], 6)
        # x_busy = xx[index_num, :]
        # x_bef = np.zeros((6, 5))
        # x_obs = np.empty((1, 5))
        # res_set = np.empty((1, 3))
        # y_obs = np.empty(0)
        # t_left = np.zeros(6)
        # iter = 0
        # t = 0
        # t_set = np.empty(0)
        # for i in range(0, 6):
        #     t_left[i] = cost_function(x_busy[i], x_bef[i]) + 20
        #
        # x1_var = np.arange(1.5, 3.01, 0.02)
        # x2_var = np.array([0.3, 0.5, 0.8])
        # x3_var = np.arange(0.1, 1.01, 0.01)
        # x4_var = np.array([0.15, 0.3, 0.5])
        # x5_var = np.concatenate([np.arange(0.25, 0.3333, 0.005), np.arange(0.3333, 0.5, 0.01), np.arange(0.5, 1, 0.015), np.arange(1, 3.03, 0.03)])
        # v = np.random.rand(6, 5)
        # v[:, 0] = (v[:, 0] - 0.5) / 2.5
        # v[:, 1] = (v[:, 1] - 0.5) / 5
        # v[:, 2] = (v[:, 2] - 0.5) / 5
        # v[:, 3] = (v[:, 3] - 0.5) / 10
        # v[:, 4] = (v[:, 4] - 0.5) / 2.5
        # w = 1.2
        # c = 2
        # opt_p = np.zeros(6)
        # opt_pl = x_busy.copy()
        # opt_g = 0
        # opt_gl = np.zeros(5)
        # vmax = np.array([0.3, 0.3, 0.2, 0.2, 0.5])
        # vmin = np.array([-0.3, -0.3, -0.2, -0.2, -0.5])
        #
        # while iter < 150:
        #     index = np.argmin(t_left)
        #     t_temp = t_left[index]
        #     t_left -= t_temp
        #     t += t_temp
        #     t_set = np.concatenate([t_set, [t]])
        #     x_obs = np.concatenate([x_obs, x_busy[index].reshape(1, -1)])
        #     res = experiment(x_busy[index].reshape(1, -1))
        #     res_set = np.concatenate([res_set, res])
        #     y_new = black_box_function(res)
        #     y_obs = np.concatenate([y_obs, y_new])
        #     if y_new > opt_p[index]:
        #         opt_p[index] = y_new
        #         opt_pl[index] = x_busy[index]
        #         if y_new > opt_g:
        #             opt_g = y_new
        #             opt_gl = x_busy[index]
        #
        #     x_bef[index] = x_busy[index]
        #     v[index] = w * v[index] + c * np.random.rand(1) * (opt_pl[index] - x_busy[index]) + c * np.random.rand(1) * (opt_gl - x_busy[index])
        #     v[index][v[index] > vmax] = vmax[v[index] > vmax]
        #     v[index][v[index] < vmin] = vmin[v[index] < vmin]
        #     candidate = x_busy[index] + v[index]
        #     index_next = np.array([0, 0, 0, 0, 0])
        #     for i in range(len(x1_var)):
        #         if x1_var[i] > candidate[0]:
        #             break
        #     index_next[0] = i
        #     for i in range(len(x2_var)):
        #         if x2_var[i] > candidate[1]:
        #             break
        #     index_next[1] = i
        #     if i >= 1 and np.random.rand(1) > 0.5:
        #         index_next[1] = i - 1
        #     for i in range(len(x3_var)):
        #         if x3_var[i] > candidate[2]:
        #             break
        #     index_next[2] = i
        #     for i in range(len(x4_var)):
        #         if x4_var[i] > candidate[3]:
        #             break
        #     index_next[3] = i
        #     if i >= 1 and np.random.rand(1) > 0.5:
        #         index_next[3] = i - 1
        #     for i in range(len(x5_var)):
        #         if x5_var[i] > candidate[4]:
        #             break
        #     index_next[4] = i
        #     x_busy[index] = np.array([x1_var[index_next[0]], x2_var[index_next[1]], x3_var[index_next[2]],
        #                               x4_var[index_next[3]], x5_var[index_next[4]]])
        #     t_left[index] = cost_function(x_busy[index], x_bef[index]) + 20
        #     print('Iteration {0} finished!'.format(iter))
        #     iter += 1
        # x_obs = np.delete(x_obs, 0, axis=0)
        # res_set = np.delete(res_set, 0, axis=0)

        # method 7: HTS (CAPBO)
        # penalization = 10
        # t_budget = 200
        # index_num = np.random.randint(0, xx.shape[0], 6)
        # x_busy = xx[index_num, :]
        # x_bef = np.zeros((6, 5))
        # x_obs = np.empty((1, 5))
        # res_set = np.empty((1, 3))
        # y_obs = np.empty(0)
        # t_left = np.zeros(6)
        # iter = 0
        # t = 0
        # t_set = np.empty(0)
        # for i in range(0, 6):
        #     t_left[i] = max(cost_function(x_busy[i], x_bef[i]), 20)
        #
        # while iter < 10:
        #     index = np.argmin(t_left)
        #     t_temp = t_left[index]
        #     t_left -= t_temp
        #     t += t_temp
        #     t_set = np.concatenate([t_set, [t]])
        #     x_obs = np.concatenate([x_obs, x_busy[index].reshape(1, -1)])
        #     res = experiment(x_busy[index].reshape(1, -1))
        #     res_set = np.concatenate([res_set, res])
        #     y_new = black_box_function(res)
        #     y_obs = np.concatenate([y_obs, y_new])
        #
        #     x_bef[index] = x_busy[index]
        #     index_num = np.random.randint(0, xx.shape[0], 1)
        #     x_busy[index] = xx[index_num, :]
        #     t_left[index] = max(cost_function(x_busy[index], x_bef[index]), 20)
        #     print('Iteration {0} finished!'.format(iter))
        #     iter += 1
        #
        # res_set = np.delete(res_set, -1, axis=0)
        # gpr_bayers1 = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-2, normalize_y=True,
        #                                        n_restarts_optimizer=20, random_state=1537 + seed)
        # gpr_bayers2 = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-2, normalize_y=True,
        #                                        n_restarts_optimizer=20, random_state=1537 + seed)
        # true_set = experiment(x_obs[-1, :].reshape(1, -1))
        # fake_set = np.zeros((1, 3))
        # fake_set[0, 0] = true_set[0, 0]
        # gpr_bayers1.fit(x_obs[:-1, :], res_set[:, 1])
        # fake_set[0, 1] = gpr_bayers1.predict(x_obs[-1, :].reshape(1, -1))
        # gpr_bayers2.fit(x_obs[:-1, :], res_set[:, 2])
        # fake_set[0, 2] = gpr_bayers2.predict(x_obs[-1, :].reshape(1, -1))
        # y_fake = black_box_function(fake_set)
        # res_set = np.concatenate([res_set, fake_set])
        # y_obs = np.concatenate([y_obs, y_fake])
        #
        # gpr_bayers = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-2, normalize_y=True,
        #                                       n_restarts_optimizer=20, random_state=1537 + seed)
        # while iter < 50:
        #     i = np.argmin(t_left)
        #     t_temp = t_left[i]
        #     t_left -= t_temp
        #     t += t_temp
        #     t_set = np.concatenate([t_set, [t]])
        #
        #     gpr_bayers.fit(x_obs, y_obs)
        #     pen = max(penalization * (1 - t_set[-1] / t_budget), 0)
        #     mean, std = gpr_bayers.predict(xx, return_std=True)
        #     a = mean - max(y_obs)
        #     z = a / std
        #     ei = a * norm.cdf(z) + std * norm.pdf(z)
        #     t_est = cost_f_true(xx, x_busy[i])
        #     pen *= np.std(ei) / np.mean(t_est)
        #     ei = a * norm.cdf(z) + std * norm.pdf(z) - pen * t_est
        #     index = np.argmax(ei)
        #     while ei[index] < 0:
        #         pen *= 0.7
        #         ei = a * norm.cdf(z) + std * norm.pdf(z) - pen * t_est
        #         index = np.argmax(ei)
        #
        #     x_new = xx[index, :]
        #     x_bef[i] = x_busy[i]
        #     x_busy[i] = x_new
        #     res_set[-1, :] = true_set
        #     y_obs[-1] = black_box_function(true_set)
        #
        #     true_set = experiment(x_new.reshape(1, -1))
        #     fake_set[0, 0] = true_set[0, 0]
        #     gpr_bayers1.fit(x_obs, res_set[:, 1])
        #     fake_set[0, 1] = gpr_bayers1.predict(x_new.reshape(1, -1))
        #     gpr_bayers2.fit(x_obs, res_set[:, 2])
        #     fake_set[0, 2] = gpr_bayers2.predict(x_new.reshape(1, -1))
        #     x_obs = np.concatenate([x_obs, x_new.reshape(1, -1)])
        #     res_set = np.concatenate([res_set, fake_set])
        #     y_fake = black_box_function(fake_set)
        #     y_obs = np.concatenate([y_obs, y_fake])
        #
        #     t += max(cost_function(x_new, x_bef[i]), 20)
        #     t_set = np.concatenate([t_set, [t]])
        #     print('Iteration {0} finished!'.format(iter))
        #     iter += 1
        #
        # x_obs = np.delete(x_obs, 0, axis=0)
        # res_set = np.delete(res_set, 0, axis=0)

        # method 7: DoE
        # xx1 = np.arange(1.5, 3.01, 0.02)
        # xx2 = np.array([0.3, 0.5, 0.8])
        # xx3 = np.arange(0.1, 1.01, 0.01)
        # xx4 = np.array([0.15, 0.3, 0.5])
        # xx5 = np.concatenate([np.arange(0.25, 0.3333, 0.005), 
        #                     np.arange(0.3333, 0.5, 0.01), 
        #                     np.arange(0.5, 1, 0.015), 
        #                     np.arange(1, 3.03, 0.03)])

        # # 将x1映射到xx1中最接近的值
        # def map_to_nearest(xx, x):
        #     nearest_vals = np.array([xx[np.abs(xx - xi).argmin()] for xi in x])
        #     return nearest_vals

        # # 拉丁方采样函数，n为采样数量
        # def latin_hypercube_sampling(n, xx):
        #     n_dims = xx.shape[1]  # 获取维度数
        #     sample = lhs(n_dims, samples=n)  # 拉丁方采样
        #     x1 = sample[:, 0] * 1.5 + 1.5
        #     x2 = sample[:, 1] * 0.5 + 0.3
        #     x3 = sample[:, 2] * 0.9 + 0.1
        #     x4 = sample[:, 3] * 0.35 + 0.15
        #     x5 = sample[:, 4] * 2.75 + 0.25
        #     # 将采样点映射到最近的离散值
        #     x1_mapped = map_to_nearest(xx1, x1)
        #     x2_mapped = map_to_nearest(xx2, x2)
        #     x3_mapped = map_to_nearest(xx3, x3)
        #     x4_mapped = map_to_nearest(xx4, x4)
        #     x5_mapped = map_to_nearest(xx5, x5)

        #     # 返回映射后的样本点
        #     return np.vstack([x1_mapped, x2_mapped, x3_mapped, x4_mapped, x5_mapped]).T

        # # 线性模型拟合
        # def fit_model(X, y):
        #     model = LinearRegression()
        #     model.fit(X, y)
        #     return model

        # # 迭代过程
        # x_bef = np.array([0, 0, 0, 0, 0])
        # t = 0
        # iter = 0
        # res_set = np.empty([0, 3])
        # t_set = np.empty(0)
        # y_obs = np.empty(0)
        
        # n_iterations = 3
        # shrink_factor = 0.5  # 空间缩小为30%
        # for iteration in range(1, n_iterations + 1):

        #     # 第一次采样，全空间采样
        #     n_samples = 10
        #     X_sample = latin_hypercube_sampling(n_samples, xx)
        #     res = experiment(X_sample)
        #     res_set = np.concatenate([res_set, res])
        #     y_sample = np.array([black_box_function(x.reshape(1, -1)) for x in res])
        #     y_obs = np.concatenate([y_obs, y_sample.reshape(-1)])
        #     for x_now in X_sample:
        #         t += cost_function(x_now, x_bef) + 20
        #         x_bef = x_now
        #         t_set = np.concatenate([t_set, [t]])
        #         iter += 1

        #     # 线性模型拟合
        #     model = fit_model(X_sample, y_sample)
        #     y_pred = model.predict(xx)

        #     # 找到当前最优点
        #     best_idx = np.argmax(y_pred)
        #     best_point = xx[best_idx]

        #     # 缩小采样空间
        #     new_ranges = []
        #     for i in range(len(best_point)):
        #         min_val = np.max([best_point[i] - shrink_factor * (np.max(xx[:, i]) - np.min(xx[:, i])), np.min(xx[:, i])])
        #         max_val = np.min([best_point[i] + shrink_factor * (np.max(xx[:, i]) - np.min(xx[:, i])), np.max(xx[:, i])])
        #         new_ranges.append(np.linspace(min_val, max_val, len(xx)))

        #     # 生成新的采样点
        #     xx1 = xx1[xx1 >= new_ranges[0][0]]
        #     xx1 = xx1[xx1 <= new_ranges[0][-1]]
        #     xx2 = xx2[xx2 >= new_ranges[1][0]]
        #     xx2 = xx2[xx2 <= new_ranges[1][-1]]
        #     xx3 = xx3[xx3 >= new_ranges[2][0]]
        #     xx3 = xx3[xx3 <= new_ranges[2][-1]]
        #     xx4 = xx4[xx4 >= new_ranges[3][0]]
        #     xx4 = xx4[xx4 <= new_ranges[3][-1]]
        #     xx5 = xx5[xx5 >= new_ranges[4][0]]
        #     xx5 = xx5[xx5 <= new_ranges[4][-1]]
        #     t1, t2, t3, t4, t5 = np.meshgrid(xx1, xx2, xx3, xx4, xx5)
        #     xx = np.concatenate([t1.reshape(-1, 1), t2.reshape(-1, 1), t3.reshape(-1, 1), t4.reshape(-1, 1), t5.reshape(-1, 1)], axis=1)

        # # 最优点和最优值
        # best_idx = np.argmax(y_sample)
        # best_point = X_sample[best_idx]
        # best_value = y_sample[best_idx]

        # print(f"最优点: {best_point}")
        # print(f"最优值: {best_value}")

        # method 8: CMA-ES
        bounds = {
            "xx1": [1.5, 3.0],  # 对应 x1 的取值范围
            "xx2": [0.3, 0.8],  # 对应 x2 的取值范围
            "xx3": [0.1, 1.0],  # 对应 x3 的取值范围
            "xx4": [0.15, 0.5], # 对应 x4 的取值范围
            "xx5": [0.25, 3.0]  # 对应 x5 的取值范围
        }
        # 自变量的初始猜测值和搜索范围
        initial_guess = [
            np.random.uniform(bounds["xx1"][0], bounds["xx1"][1]),  # 在 xx1 范围内随机生成初始值
            np.random.uniform(bounds["xx2"][0], bounds["xx2"][1]),  # 在 xx2 范围内随机生成初始值
            np.random.uniform(bounds["xx3"][0], bounds["xx3"][1]),  # 在 xx3 范围内随机生成初始值
            np.random.uniform(bounds["xx4"][0], bounds["xx4"][1]),  # 在 xx4 范围内随机生成初始值
            np.random.uniform(bounds["xx5"][0], bounds["xx5"][1])   # 在 xx5 范围内随机生成初始值
        ]
        sigma = 0.5  # 初始搜索步长

        # CMA-ES的实例化
        es = cma.CMAEvolutionStrategy(initial_guess, sigma, {'bounds': [list(bounds[key][0] for key in bounds), list(bounds[key][1] for key in bounds)]})


        # 优化过程
        x_bef = np.array([0, 0, 0, 0, 0])
        t = 0
        iter = 0
        res_set = np.empty([0, 3])
        t_set = np.empty(0)
        y_obs = np.empty(0)
        max_iterations = 20  # 最大迭代次数
        for generation in range(max_iterations):
            # 1. 生成新的候选解
            solutions = es.ask()

            # 2. 评估候选解的目标函数值 (负值是因为CMA-ES默认是最小化，我们要最大化目标函数)
            fitnesses = [-black_box_function(experiment(x.reshape(1, -1))) for x in solutions]

            res = experiment(np.concatenate([*solutions]).reshape(-1, 5))
            res_set = np.concatenate([res_set, res])
            y_sample = np.array([black_box_function(x.reshape(1, -1)) for x in res])
            y_obs = np.concatenate([y_obs, y_sample.reshape(-1)])
            for x_now in solutions:
                t += cost_function(x_now, x_bef) + 20
                x_bef = x_now
                t_set = np.concatenate([t_set, [t]])
                iter += 1

            # 3. 更新CMA-ES模型
            es.tell(solutions, fitnesses)

            # 4. 打印当前迭代的信息
            print(f"Generation {generation}: Best solution so far {es.best.x}, Fitness: {-es.best.f}")

            # 停止条件（例如目标函数值变化不大）
            if es.stop():
                break

            # 输出最终结果
            best_solution = es.best.x
            best_fitness = -es.best.f  # 恢复最大化问题

            print(f"\n最优解: {best_solution}")
            print(f"最优值: {best_fitness}")

        df = pd.read_excel('sensitivity_4dim_time.xlsx', header=0)
        df[seed] = t_set
        pd.DataFrame(df).to_excel('sensitivity_4dim_time.xlsx', index=False, header=True)
        df = pd.read_excel('sensitivity_4dim_result.xlsx', header=0)
        df = pd.concat([df, pd.DataFrame(res_set, columns=['I' + str(seed), 'FE' + str(seed), 'y' + str(seed)])], axis=1)
        df = pd.concat([df, pd.DataFrame(y_obs, columns=['obj' + str(seed)])], axis=1)
        pd.DataFrame(df).to_excel('sensitivity_4dim_result.xlsx', index=False, header=True)
