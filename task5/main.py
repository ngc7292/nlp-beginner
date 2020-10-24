import fitlog
import random
import argparse

# 从命令行传入参数
# parser = argparse.ArgumentParser()
# parser.add_argument('--demo', type=int, default=2)

fitlog.commit(__file__)             # 自动 commit 你的代码
fitlog.set_log_dir("logs/")         # 设定日志存储的目录

# args = parser.parse_args()
# fitlog.add_hyper(args)  # 通过这种方式记录ArgumentParser的参数
hyper_data = {'random':1}
fitlog.add_hyper(hyper_data)

# fitlog.add_hyper_in_file(__file__)  # 记录本文件中写死的超参数

######hyper
rand_seed = 124
######hyper

random.seed(rand_seed)
best_acc, best_step, step = 0, 0, 0

for i in range(200):
    step += 1
    if step % 20 == 0:
        loss = random.random()
        acc = random.random()
        fitlog.add_loss(loss,name="Loss",step=step)
        fitlog.add_metric({"dev":{"Acc":acc}}, step=step)
        if acc>best_acc:
            best_acc = acc
            fitlog.add_best_metric({"dev":{"Acc":best_acc}})
            # 当dev取得更好的performance就在test上evaluate一下
            test_acc = random.random()
            fitlog.add_best_metric({"test":{"Acc":test_acc}})
fitlog.finish()                     # finish the logging                 # finish the logging