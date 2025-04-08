基于[linyiLYi/snake-ai](https://github.com/linyiLYi/snake-ai/blob/master/README_CN.md) 做的改进, 
主要思路：
1. 增加额外状态对下一个方向是否安全给出提示，是否安全的判断标准是看蛇头到蛇尾巴是否存在通路，同时如果存在通路，奖励加倍，惩罚减半。
2. 第一步改造完之后已经可以轻松在12*12的board上把蛇长度跑到120+，不过发现经常蛇头紧追着蛇尾跑，导致有些果子吃不到，因此在连通性基础上鼓励蛇头和蛇尾保持一定距离

## 测试效果

board的大小是12*12的，snake初始长度为3，最大144个，每吃一个果子score+10。   
测试下来蛇的长度可以比较容易增长到135+，25%~30%概率可以填满。  
下面是一次测试的输出:
```text
$ python test_multi_input.py > test_multi_input.log
$ cat test_multi_input.log |  grep 'Total Steps' | grep '14\d$'
Episode 1: Reward Sum: 149.8683, Score: 1410, Total Steps: 3870, Snake Size: 144
Episode 2: Reward Sum: 150.5437, Score: 1410, Total Steps: 3780, Snake Size: 144
Episode 3: Reward Sum: 136.1486, Score: 1390, Total Steps: 3877, Snake Size: 142
Episode 4: Reward Sum: 135.9879, Score: 1390, Total Steps: 4858, Snake Size: 142
Episode 5: Reward Sum: 151.8852, Score: 1410, Total Steps: 3460, Snake Size: 144
Episode 6: Reward Sum: 156.0510, Score: 1410, Total Steps: 4718, Snake Size: 144
Episode 8: Reward Sum: 154.7698, Score: 1410, Total Steps: 3674, Snake Size: 144
Episode 10: Reward Sum: 132.4221, Score: 1390, Total Steps: 4872, Snake Size: 142
Episode 11: Reward Sum: 153.2695, Score: 1410, Total Steps: 3397, Snake Size: 144
Episode 13: Reward Sum: 135.3217, Score: 1370, Total Steps: 5470, Snake Size: 140
Episode 14: Reward Sum: 135.7749, Score: 1390, Total Steps: 4612, Snake Size: 142
Episode 18: Reward Sum: 139.1064, Score: 1390, Total Steps: 4621, Snake Size: 142
Episode 19: Reward Sum: 136.4951, Score: 1390, Total Steps: 4802, Snake Size: 142
Episode 25: Reward Sum: 132.8076, Score: 1370, Total Steps: 4838, Snake Size: 140
Episode 27: Reward Sum: 151.8601, Score: 1410, Total Steps: 4582, Snake Size: 144
Episode 29: Reward Sum: 149.6423, Score: 1410, Total Steps: 4712, Snake Size: 144
Episode 30: Reward Sum: 138.8079, Score: 1390, Total Steps: 3468, Snake Size: 142
```
共跑了30次，140以上的有17次，其中8次占满整个board。
注意：测试有一定的随机性，其他的测试可能不如上述结果稳定，不过跑到135+还是容易的。

### 一些细节
1. 补充状态后，怎么把这些数据和图像数据融合一块输入到神经网络  
   采用的MultiInputPolicy，可以参考[stable-baselines3的文档](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#multiple-inputs-and-dictionary-observations
) 或者直接看看项目中怎么用的。
2. 训练时候采用了DummyVecEnv,相比SubprocVecEnv要快一些，原因可以参考[stable-baseline3的tutorial](https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/3_multiprocessing.ipynb)
3. 升级了gym依赖到gymnasium==1.0.0, SnakeEnv需要稍作调整，具体所做修改可以参考项目代码。

### 训练、测试

1. 参考install_on_linux.sh、install_on_mac.sh 安装相关以来， 仅在python3.10 版本下测试过。
2. 直接运行train_multi_input.py或test_multi_input.py即可。
