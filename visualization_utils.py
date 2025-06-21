import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot_training_results(trainer, output_dir):
    """绘制训练结果图表"""
    history = trainer.state.log_history
    
    # 提取训练和评估指标
    train_loss = [x['loss'] for x in history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    eval_acc = [x['eval_accuracy'] for x in history if 'eval_accuracy' in x]
    
    # 设置中文字体
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    rcParams['axes.unicode_minus'] = False
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='训练损失')
    plt.plot(eval_loss, label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('步骤')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(eval_acc, label='验证准确率')
    plt.title('验证准确率')
    plt.xlabel('步骤')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'))
    plt.close()

def plot_comparison(results, save_path=None):
    """绘制模型对比图表"""
    models = list(results.keys())
    metrics = ["accuracy", "macro_f1", "speed", "memory_usage"]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        values = [results[model][metric] for model in models]
        ax.bar(models, values)
        ax.set_title(metric)
        
        # 添加数值标签
        for j, v in enumerate(values):
            ax.text(j, v, f"{v:.2f}", ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.savefig("model_comparison.png")  # 默认保存路径
    plt.close()
