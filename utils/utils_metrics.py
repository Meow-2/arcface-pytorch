import numpy as np
import torch
from scipy import interpolate
from sklearn.model_selection import KFold
from tqdm import tqdm


# 分成 10 折, train:test = 9:1, 最终返回的是10个测试集上的Acc的列表
# 对于每个测试集上的阈值选取, 由其对应的训练集上的最佳阈值来确定
# 在训练集上, 在每个阈值下计算 Acc, 取 Acc 最高作为最佳阈值
# 然后用该阈值, 计算测试集上的 Acc, 作为该折的 Acc

# 分成 10 折, train:test = 9:1, 最终返回的是10个测试集上的 Validation Rate 的均值、方差和 Far 的均值
# 在训练集上, 选取 FAR = 0.001 的阈值, 放在测试集上, 取得测试集的 Val 和 Far
# 每个折取均值

# tpr,fpr 用来画 ROC 曲线 (400)
# accuracy 是准确率评价指标 (10)
# val 是每个折上的均值
# val_std 是每个折上的标准差
# far 是每个折上的均值
def evaluate(distances, labels, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    # thresholds = np.ones(400)*0.5
    # accuracy 是一个 (10) 的数组, 带表每一折测试集划分下, 最佳阈值的测试集上的准确率
    # 划分成10折是为了计算均值和方差

    # (400), (400), (10), 第 10 折的最佳阈值(没有什么意义)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, distances,
                                                        labels, nrof_folds=nrof_folds)
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.001)
    # thresholds = np.ones(400)*0.5
    val, val_std, far = calculate_val(thresholds, distances,
                                      labels, 1e-2, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far, best_thresholds
    #     (400),(400), (10),   (1),   (1),   (1),  (1)


def calculate_roc(thresholds, distances, labels, nrof_folds=10):
    # thresholds = np.arange(0, 4, 0.01)
    nrof_pairs = min(len(labels), len(distances))  # 总共检测的人脸对对数
    nrof_thresholds = len(thresholds)  # 阈值个数, 400 个
    # 10折, 不打乱, 也就是划分成10份数据, 每一份轮流作为验证集
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    # (10, 400)
    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    # (10)
    accuracy = np.zeros((nrof_folds))
    # 指数, 索引, 每一个 pair 的索引
    indices = np.arange(nrof_pairs)

    # train_set 和 test_set 是索引列表
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        # 记录每个阈值下的准确率 (400)
        acc_train = np.zeros((nrof_thresholds))
        # 遍历阈值, 记录每个阈值下训练集的准确率
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, distances[train_set], labels[train_set])
        # 记录 accuracy 最高的阈值的索引
        best_threshold_index = np.argmax(acc_train)

        # 遍历阈值, 记录每个阈值下每个fold 、每个阈值下测试集的 tpr 和 fpr
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, distances[test_set], labels[test_set])

        # 计算测试集上的准确率,
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], distances[test_set], labels[test_set])
        # 这里需要注意
        # tpr 和 fpr 只有最后一个folder计算的才是有用的, 之前的都是没有用的
        # np.mean(tprs, 0) 并不改变 tprs 的内容
        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        # 所以最后的 tpr 和 fpr 才是返回的有用的值, 之前的都是白算了
        # 最后一个 fold 会填满 tprs 和 fprs 数组, 然后沿着 fold 的轴求均值, 也就是说会返回一个 (400) 的数组
        # 为每个阈值下, 每个 fold 的平均值
    return tpr, fpr, accuracy, thresholds[best_threshold_index]
    # (400), (400), (10), 第 10 折的最佳阈值(没有什么意义)


def calculate_accuracy(threshold, dist, actual_issame):
    # np.less(x1,x2) 对两个数组中对应位置的元素进行比较, 如果 x1 中的元素小于 x2 的元素, 则返回 True
    # 所有的距离与阈值比较
    predict_issame = np.less(dist, threshold)
    # 计算 tp fp tn fn
    # 判断两个人是同一个人, 实际上两个人是同一个人
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    # 判断两个人是同一个人, 实际上两个人不是同一个人
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    # 判断两个人不是同一个人, 实际上两个人不是同一个人
    tn = np.sum(np.logical_and(np.logical_not(
        predict_issame), np.logical_not(actual_issame)))
    # 判断两个人不是同一个人, 实际上两个人是同一个人
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn == 0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn == 0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
    # pair 的数量
    nrof_pairs = min(len(labels), len(distances))
    # 阈值的数量
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    # (10)
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        # 遍历阈值, 计算每个阈值下的 val 和 far
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, distances[train_set], labels[train_set])
        # 绘制函数
        if np.max(far_train) >= far_target:
            # 这里有问题, 还不清楚是什么问题
            # 一维插值函数, 将 far_train 映射到每个阈值
            # far_train 到 thresholds 有一个映射关系
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            # 选取 far_target = 0.001 对应的阈值
            threshold = f(far_target)
        else:
            threshold = 0.0

        # 计算这个阈值下, 这个 fold 的 val 和 far
        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, distances[test_set], labels[test_set])

    # 每个 fold 的平均值
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean  # far 的均值


def calculate_val_far(threshold, dist, actual_issame):
    # np.less(x1,x2) 对两个数组中对应位置的元素进行比较, 如果 x1 中的元素小于 x2 的元素, 则返回 True
    # 所有的距离与阈值比较
    predict_issame = np.less(dist, threshold)
    # tp
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    # fp
    false_accept = np.sum(np.logical_and(
        predict_issame, np.logical_not(actual_issame)))
    # tp + fn
    n_same = np.sum(actual_issame)
    # tn + fp
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0, 0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def test(test_loader, model, png_save_path, log_interval, batch_size, cuda):
    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        with torch.no_grad():
            # --------------------------------------#
            #   加载数据，设置成cuda
            # --------------------------------------#
            data_a, data_p = data_a.type(
                torch.FloatTensor), data_p.type(torch.FloatTensor)
            if cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()
            # --------------------------------------#
            #   传入模型预测，获得预测结果
            #   获得预测结果的距离
            # --------------------------------------#
            out_a, out_p = model(data_a), model(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))

        # --------------------------------------#
        #   将结果添加进列表中
        # --------------------------------------#
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        # --------------------------------------#
        #   打印
        # --------------------------------------#
        if batch_idx % log_interval == 0:
            pbar.set_description('Test Epoch: [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    # --------------------------------------#
    #   转换成numpy
    # --------------------------------------#
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(
        distances, labels)
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    # print('Best_thresholds: %2.5f' % best_thresholds)
    # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, 1e-2))
    plot_roc(fpr, tpr, figure_name=png_save_path)


# 画 ROC 图, 根据每个阈值下的 fpr 和 tpr
def plot_roc(fpr, tpr, figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    # 用来计算 ROC 面积
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)
