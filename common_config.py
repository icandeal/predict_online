Accuracy = 0.2   # 关联样本中的分布概率
Sample_num = 20  # 关联样本数量
increase_rate = 0.3  # 关联样本中的分布概率相对于知识点掌握程度的自然分布概率的提升率
accuracy_adjust = [0.0, 0.0, 0.0, 0.0]  # 对样本分标签种类调节分布概率阀值
step = 5  # 准确度的分段步长
rule_main = str(Accuracy)+'_'+str(Sample_num)+'_'+str(increase_rate)+'_0.0_0.0_0.0_0.0'
rule = str(Accuracy)+'_'+str(Sample_num)+'_'+str(increase_rate)+'_'+str(accuracy_adjust[0])\
       + '_'+str(accuracy_adjust[1])+'_'+str(accuracy_adjust[2])+'_'+str(accuracy_adjust[3])
