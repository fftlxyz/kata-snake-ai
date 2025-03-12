基于[linyiLYi/snake-ai](https://github.com/linyiLYi/snake-ai/blob/master/README_CN.md) 做的改进, 
主要思路：增加额外状态对下一个方向是否安全给出提示，是否安全的判断标准是看蛇头到蛇尾巴是否存在通路，同时如果存在通路，奖励加倍，惩罚减半。

### 测试效果

board的大小是12*12的，snake初始长度为3，最大144个，每吃一个果子score+10。   
测试下来蛇的长度可以比较容易增长到120，运气好可以把整个盘子填满。  
下面是一次测试的输出:
```text
$ python test_multi_input.py > test_multi_input.log
$ cat test_multi_input.log | grep 'Total Steps'

Episode 1: Reward Sum: 164.9171, Score: 1410, Total Steps: 3770, Snake Size: 144
Episode 2: Reward Sum: 148.4114, Score: 1390, Total Steps: 4003, Snake Size: 142
Episode 3: Reward Sum: 114.3453, Score: 1210, Total Steps: 3107, Snake Size: 123
Episode 4: Reward Sum: 129.5856, Score: 1290, Total Steps: 4157, Snake Size: 132
Episode 5: Reward Sum: 129.9322, Score: 1300, Total Steps: 3192, Snake Size: 132
Episode 6: Reward Sum: 101.3748, Score: 1130, Total Steps: 2665, Snake Size: 116
Episode 7: Reward Sum: 144.4728, Score: 1370, Total Steps: 3828, Snake Size: 140
Episode 8: Reward Sum: 164.3303, Score: 1410, Total Steps: 4321, Snake Size: 144
Episode 9: Reward Sum: 136.6518, Score: 1330, Total Steps: 4466, Snake Size: 136
Episode 10: Reward Sum: 109.1173, Score: 1170, Total Steps: 2947, Snake Size: 120
```
注意：测试有一定的随机性，其他的测试可能不如上述结果稳定，不过10次中有几次跑到120+还是容易的。

### 一些细节
1. 补充状态后，怎么把这些数据和图像数据融合一块输入到神经网络  
   采用的MultiInputPolicy，可以参考[stable-baselines3的文档](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#multiple-inputs-and-dictionary-observations
) 或者直接看看项目中怎么用的。
2. 训练时候采用了DummyVecEnv,相比SubprocVecEnv要快一些，原因可以参考[stable-baseline3的tutorial](https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/3_multiprocessing.ipynb)
3. 升级了gym依赖到gymnasium==1.0.0, SnakeEnv需要稍作调整，具体所做修改可以参考项目代码。

### 训练、测试

1. 参考install_on_linux.sh、install_on_mac.sh 安装相关以来
2. 直接运行train_multi_input.py或test_multi_input.py即可。
