from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time
import kernels
import Lovasz_losses_tf as L

'''
TensorFlow不单独地运行单一的复杂计算,而是让我们可以先用图描述一系列可交互的计算操作，然后全部一起在Python之外运行

首先自己的数据按照规定的方式进行预处理，接着将预处理后的数据划分，分成训练集、验证集。测试集
所有的数据都分别视同该规定的生成器方法生成，因为内存限制和优化的目的，无法直接把整个数据集一次输进去
所以分批次输入，相当于给数据打包成一个一个部分
假设数据在做左边的水池，中间是横纵交错的管道，每个交叉道口都有一个开关，右边是一个新的水池，叫做分类
数据分批，作为输入的数据源，session是总闸，op是运算器，session开启后，op中的参数运算，也就表示数据就可以从左边的水池流入右边的水池，中间管道的开关就是激活函数处理的线性函数，负责开关
tf中在初始化管道时，会通过迭代器函数（自己定调构建，不像反向传播，是session开启后，内部直接运算的）不断把打包好的数据输入管道，直到数据都输入停止

训练集按照批次输入，第一个批次结束，得到逻辑值，也就是一个(b*n)*c的二维矩阵，c其实就是特征，长度等于class类个数
标签开始是(b*n)*1,使用热点编码修改成(b*n)*c，这样就和逻辑值一样了
这个逻辑值和标签进行计算loss，使用tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)函数运算的
第一步是先对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率，也就是将特征参数进行归一化(0-1之间)
第二步是softmax的输出向量[c1，c2,c3...](19个)和样本的实际标签做一个交叉熵，样本的实际标签为[0 0 1 0 0 ...](19个)，这是热点编码处理，方便运算
预测越准确，结果的值越小（别忘了前面还有负号）
注意！！！这个函数的返回值并不是一个数，而是一个向量，如果要求交叉熵，我们要再做一步tf.reduce_sum操作,就是对向量里面所有元素求和
如果求loss，则要做一步tf.reduce_mean操作，对向量求均值
当然，模型其实一开始不是为了找到属于那个标签，只是生硬的将特征归一化，而损失函数loss才是用来约束最后生成的特征的
优化器可以选择loss下降最快的梯度运算，加速运算，loss相当于限制条件，真正结果和预测结果之间的差值
如果loss为0，就代表着预测概率[0 0 1 0 0 ...]和标签值[0 0 1 0 0 ...]一样，现实运算很难为0
而且这里也不是直接选择概率最大的当作标签，因为要迭代训练，直到loss很小，也就是找到的确实是最大概率，也就是对应的标签越准确
在验证集中反而会将归一化特征直接变成[0 0 1 0 0 ...]这种形式，是为了进行评估
loss减小也就是要求最后生成的归一化特征和标签对应，反向传播平摊误差，然后第二个批次输入计算，这种mini_batch的方式，好处多多
mini_batch可以作为一种优化算法，因为很难一次将数据输入（内存太大）
因为第二个批次和第一个批次不一样，这样每次计算，虽然优化器显示的朝着最小化loss方向，但是变化不大，有的还增加
这是因为这些迭代只是完成了一个epoch（一个epoch代表数据集完完整整输入了一次），所以差不多，但是当进行下一个epoch，可以看到loss有了变化
虽然损失函数loss可能增加也可能减少，但是总体方向是朝着最低处走的

mini_batch之间可能有重叠，但是用的是概率选中心点，所以重叠很小
6个batch_size，第一个生成后，第二个也是是相同的方法
而且选取中心点后，用的是随机排序，这样可以减少相邻点云空间位置对模型的影响，提高模型的泛化能力
而且第二个epoch和第一个epoch也不同，这是因为概率也是随机生成的
这样，两个随机就可以增强模型的泛化能力，而且也解决了点云无序性的问题，不规则特征是KNN解决的，感觉kNN很像是kpconv网络中的可变形卷积方法
这也就是为啥，第二个epoch的损失函数loss相对于第一个epoch可能增加也可能减少，但是总体方向是朝着最低处走的
增加和减少是因为随机后新的batch构成的空间几何结构和特征变化了，可能不好学，也可能更好学了（猜测是几何特征结构相似度很高）
总体朝着最低处，是因为epoch迭代下去，学的差不多了，越来越好
规定一个epoch迭代的最大次数，是因为loss很难降到0，而且越到后面，学习的越慢，梯度衰减的越难，没有必要了，而且还会过拟合

一般只要所有优化的参数都设置一样，影响最后语义分割的精度的问题，最大的是模型中间的提取结构
所以可以优化的方法有：网络的层数，一般都差不多，太深了过拟合，太浅了欠拟合，参数过多计算的也慢
                 一次性输入的点数，40960，这是计算机的内存决定的，换服务器跑，但是作模型对比时，要放在相同环境下运算对比，否则不公平
                  损失函数loss
                  其他的，优化器，采样，等等等等
所以要兼顾计算成本、运算时间，也就是提到的既要效率高又要高效
效率高是算的快，高效是算的准        

训练完一个epoch后计算交并比等参数，在一个epoch中，每一步都计算准确度，这个准确度是训练集的准确度
交并比是使用验证集计算出的交并比
训练集训练完一个epoch，验证集也训练了一个完整的epoch，验证集和训练集的随机方式是一样的，这样验证集才有验证效果
可以真实的评价一个模型的优劣 

dataset是总数据集，调用不同的参数可以分别用于训练集和验证集

只要steps*batch_size*num_points>数据集总点数，就可以全覆盖
所以batch_size*num_points内存限制时，增加steps数量，虽然效果不一定比增大batch_size*num_points好
'''


def weight_variable(shape):
    # tf.set_random_seed(42)
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2 / shape[-1]))
    initial = tf.round(initial * tf.constant(1000, dtype=tf.float32)) / tf.constant(1000, dtype=tf.float32)
    return tf.Variable(initial, name='weights')


def log_out(out_str, f_out):  # 定义一个日志输出函数，内容是输出的字符和输出的路径
    f_out.write(out_str + '\n')  # 使用写函数将会字符串写进去并回车
    f_out.flush()  # 用来刷新缓冲区
    print(out_str)  # 在运行窗口打印输出的字符 例如：Step 00000050 L_out=17.793 Acc=0.29 --- 1256.64 ms/batch


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs  # 数据平铺，将数据集的参数展开，迭代循环的
        self.config = config  # 文件路径
        # Path of the result folder
        if self.config.saving:  # 如果ture保存文件
            if self.config.saving_path is None:  # 如果文件路径不存在
                # 构建文件路径，time.strftime将时间转换成str形式
                self.saving_path = time.strftime('/media/hello/76FCB52FFCB4EB0F1/data/Labelled/results/Log_%Y-%m-%d_%H-%M-%S',
                                                 time.gmtime())
            else:
                self.saving_path = self.config.saving_path  # 路径存在
            makedirs(self.saving_path) if not exists(self.saving_path) else None  # 判断路径是否存在，不存在就创建路径os.makedirs

        with tf.variable_scope('inputs'):  # 定义创建变量（层）操作的上下文管理器
            self.inputs = dict()  # 定义字典，下面进行了字典的构建
            num_layers = self.config.num_layers  # 网络层数 num_layers=4
            # 为 input字典中添加参数，前四个参数是包含所有层次的数据，后四个参数是原始输入的数据一个batch的表示内容
            self.inputs['xyz'] = flat_inputs[:num_layers]  # 前四列是4层的[B*N*3]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]  # 4-7是四层的邻居点索引[B*N*k]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]  # 8-11是四层采样后的索引[B*N1*k]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]  # 12-15是四层插值（上采样）索引[B*N*1]
            self.inputs['features'] = flat_inputs[4 * num_layers]  # 16是特征 B*N*3
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]  # 17是标签 B*N*1
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]  # 18是输入的编号 B*N*1
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]  # 19是云标号 B*N*1

            self.labels = self.inputs['labels']  # 获取标签值
            # 此时并没有把要输入的数据传入模型，它只会分配必要的内存。
            # 等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据
            self.is_training = tf.placeholder(tf.bool, shape=())  # 占位符，在sesson中传入是否训练，训练集训练True，验证集为不训练Flase
            self.training_step = 1  # 训练步数从1开始
            self.training_epoch = 0  # 训练epoch从零开始
            self.correct_prediction = 0  # 正确预测从0开始
            self.accuracy = 0  # 准确度从0开始
            self.mIou_list = [0]  # 平均交并比从0开始
            self.loss_type = 'lovas'  # wce, lovas
            self.class_weights = DP.get_class_weights(dataset.num_per_class, self.loss_type)
            print(self.class_weights)
            self.Log_file = open(
                'log_train_' + dataset.name + str(dataset.val_split) + '.txt', 'a')
            # 打开日志输出文件,a是追加，不存在就创建

        with tf.variable_scope('layers'):  # 定义创建变量（层）操作的上下文管理器
            self.logits = self.inference(self.inputs, self.is_training)  # b*n*c  全连接层（往往是模型的最后一层）的值，一般代码中叫做logits
        # 获取逻辑值,这里只是获得了一个 b*n*c的张量，还不是分类结果，只是特征维度设置成c，方便后面的运算
        # 因为在使用交叉熵损失函数时，会用到这归一化等操作，这里直接写逻辑值，是为了计算别的还要用到

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss # 计算损失函数是要求删除无效的点
        # 在计算损失函数之前要先删除无用点 loss是以一个batch为组计算的，也就是有步长steps个loss值
        #####################################################################
        with tf.variable_scope('loss'):
            # 这两句就是将逻辑值和标签重新变成一个批次，方便计算loss
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])  # b*n*c--》（b*n）*c（19）
            self.labels = tf.reshape(self.labels, [-1])  # （b*n）

            # Boolean mask of points that should be ignored 布尔掩码的点应该被忽略
            ignored_bool = tf.zeros_like(self.labels,
                                         dtype=tf.bool)  # 创建一个和标签维度一样的都是0的张量，返回将所有元素设置为零的张量，dtype=tf.bool表明设置为0变成flase
            for ign_label in self.config.ignored_label_inds:
                # equal就是判断，x, y 是不是相等，它的判断方法不是整体判断，而是逐个元素进行判断，如果相等就是True，不相等，就是False。
                # tf.logical_or 逻辑或，一真一假也为真，tf.logical_and 逻辑与，相同才判断为真
                # 0 0 = true ---》+ False----》 tf.logical_or-----》true
                ignored_bool = tf.logical_or(ignored_bool,
                                             tf.equal(self.labels, ign_label))  # （True False False False False...）

            # Collect logits and labels that are not ignored 收集不被忽略的日志和标签
            # tf.logical_not  True变成False，False变成True
            # tf.where 返回condition中值为True的位置的Tensor
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))  # 获取有效的点索引
            # tf.gather(params,indices,axis=0 ) 从params的axis维根据indices的参数值获取切片
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            print(valid_logits.shape)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape 在logit形状范围内减少标签值
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)  # 生成列表（0-19）
            inserted_value = tf.zeros((1,), dtype=tf.int32)  # 生成 [0]
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]],
                                          0)  # 0维度上拼接
            valid_labels = tf.gather(reducing_list, valid_labels_init)  # 有效标签值

            self.loss = self.get_loss(valid_logits, valid_labels,
                                      self.class_weights)  # 损失
        # 获取loss

        with tf.variable_scope('optimizer'):
            # trainable=False表示不把学习率添加到可以训练的变量中
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            # Adam即Adaptive Moment Estimation（自适应矩估计），是一个寻找全局最优点的优化算法，引入了二次梯度校正
            # minimize(self.loss)通过更新 var_list 添加操作以最大限度地最小化 loss。
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 更新变量并取出变量
        # 优化器更新变量，最小化loss

        with tf.variable_scope('results'):
            # tf.nn.in_top_k 每个样本中前K个最大的数里面（序号）是否包含对应target中的值
            # 也就是判断logits中最大值的下标和labels的是不是一样对应
            # 返回值是bool类型
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)  # 正确预测
            # tf.cast将x的数据格式转化成dtype数据类型
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))  # 正确率
            self.prob_logits = tf.nn.softmax(self.logits)  # 用于归一化得到概率模型 # b*n*c  归一化的值，含义是属于该位置的概率，一般代码叫做probs
            # prob_logits是用来计算验证集数据的，原理和交叉熵损失函数的第一步一样，都是使用softmax函数进行归一化

            # 将summary全部保存到磁盘，以便tensorboard显示
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
        # 返回等待统计的结果 准确度 归一化概率

        # 这是tf的运算配置部分
        # GLOBAL_VARIABLES: 该collection默认加入所有的Variable对象，并且在分布式环境中共享
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)  # 将最近的100个训练模型保存，100的原因是100个epoch
        c_proto = tf.ConfigProto()  # tf.ConfigProto()主要的作用是配置tf.Session的运算方式，比如gpu运算或者cpu运算
        c_proto.gpu_options.allow_growth = True  # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片。
        self.sess = tf.Session(config=c_proto)  # 启动模型
        self.merged = tf.summary.merge_all()  # 可以将所有summary全部保存到磁盘，以便tensorboard显示
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)  # 指定一个文件用来保存图,train_log文件
        self.sess.run(tf.global_variables_initializer())  # 所有变量初始化

    # 这里传入的是一个batch的数据

    def inference(self, inputs, is_training):

        d_out = self.config.d_out  # 输出的特征维度 d_out = [16, 64, 128, 256]
        feature = inputs['features']  # 特征   B*N*3 在kitti中，输入的特征就是点云的坐标
        # 首先对输入的特征进行维度的扩张，统一到8维，randla-net编码-解码的输入部分
        feature = tf.layers.dense(feature, 8, activation=None,
                                  name='fc0')  # tf.layers.dense全连接层，相当于添加一个层,8是输出的维度大小，改变inputs的最后一维，不使用激活函数
        feature = tf.nn.leaky_relu(
            tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))  # 批次归一化（正则化）后使用了激活函数
        # tf.layers.batch_normalization 以 batch 为单位进行操作，减去 batch 内样本均值，除以 batch 内样本的标准差，最后进行平移和缩放，其中缩放参数和平移参数都是可学习的参数
        # axis的值取决于按照input的哪一个维度进行BN，例如输入为channel_last format，即[batch_size, height, width, channel]，则axis应该设定为4，如果为channel_first format，则axis应该设定为-1.
        # momentum的值用在训练时，滑动平均的方式计算滑动平均值moving_mean和滑动方差moving_variance
        # training表示模型当前的模式，如果为True，则模型在训练模式，否则为推理模式。要非常注意这个模式的设定，这个参数默认值为False。
        # 如果在训练时采用了默认值False，则滑动均值moving_mean和滑动方差moving_variance都不会根据当前batch的数据更新，这就意味着在推理模式下，均值和方差都是其初始值，
        # 因为这两个值并没有在训练迭代过程中滑动更新。
        # 在训练时，我们可以计算出batch的均值和方差，迭代训练过程中，均值和方差一直在发生变化。但是在推理时，均值和方差是固定的，它们在训练过程中就被确定下来。
        # 推荐在Conv层或FC层之后，非线性变换激活层之前插入BN层
        feature = tf.expand_dims(feature, axis=2)  # 维度扩张，这里是对第二维度进行扩张  B*N*8————》B*N*1*8

        # ###########################Encoder############################
        f_encoder_list = []  # 构建特征编码列表
        for i in range(self.config.num_layers):  # 每一层循环 num_layers=4
            # lsa编码层，首先是一个膨胀残差网络
            # d_out[i]在这里定义了第i层的输出的维度
            f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            # 我的
            density_parameter = self.config.sub_sampling_ratio[i]
            # f_encoder_i = self.residual_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
            #                                   'Encoder_layer_' + str(i), density_parameter, is_training)
            # 编码层时，先进行的是特征维度扩张，后进行的是下采样
            # 接着是一个下采样层
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])  # 采样层，输入和输出只有N发生了变化，N-->N1
            feature = f_sampled_i  # 将采样后的特征赋予新的特征  # B*N1*1*D
            if i == 0:  # 第一层
                f_encoder_list.append(f_encoder_i)  # 将特征放到特征列表中
            f_encoder_list.append(f_sampled_i)  # 后面放的就是采样后的特征，加上原始层，一共有5层，也就是列表中有5个元素，每个元素的维度逐渐扩张，8--32--128-256-512
        # ###########################Encoder############################
        # 四个编码层后，解码层的1层，输出相同维度(512)的特征，[1,1]的卷积相当于mlp，B*N1*1*d
        # helper_tf_util.conv2d第一个参数输入的是特征，第二个参数是输出的维度，所以会添加get_shape()[3].value
        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)
        # 相当于一个过渡阶段
        # ###########################Decoder############################
        f_decoder_list = []  # 构建特征解码列表
        for j in range(self.config.num_layers):  # 每一层循环 num_layers=4  j从0开始，所以倒排列时-1
            # 解码层先进行上采样
            # interp_idx记录的索引号是一对多的，也就是interp_idx中有很多一样的索引，这样就可以在feature中重复获取特征，相当于增加了点
            # 假设，这里（feature）2个点{[123]、[456]}，插值（interp_idx）是4个点，就是4次搜索，
            # 索引号假设是{[0][1][0][0]}，代表了与之最近的点的索引号，那么插值后就是feature--》{[123]、[456]、[123]、[123]}，进行了重复，点数恢复了，但是特征重复了，学习到的特征还是太少了，所以才需要残差连接和mlp操作
            # U-net网络中，恢复的是已经存在的点,其实是复制特征直到点云个数满足上采样个数要求
            # 就是按照interp_idx点云的序号在feature中查找
            # 然后是mlp反卷积减少特征维度
            f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][
                -j - 1])  # 上采样 最邻近插值 B*N/256*1*d---》B*N/64*1*d d=512
            # 反卷积，这里第一个参数使用了全连接，将相同点云数量（上采样后就相同了N/64）对应的编码层点云和解码层点云拼接起来，拼接维度是第四维，也就是特征维度
            # 可以认为是对特征进行了扩张，将编码-解码的特征放到了一起，也就是N/64个（512+256）维度的特征，对于N/64个512来说，其实只有N/256个512特征是有用的，其余的都是从这里复制过来的
            # 所以采用跳跃连接有利于获得更多的特征，N/64个256维编码层特征每个都是不重复的，拼接起来后每一行都不同，mlp之后，可以学习到更更多的特征
            # f_encoder_list[-j - 2].get_shape()[-1].value=256   维度的变化d：512--》256
            # 反卷积用于上采样，这里有一个最邻近上采样了，所以这里是为了特征降维，也有上采样的作用
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                          f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)
            feature = f_decoder_i  # 将采样后的特征赋予新的特征  # B*N*1*D
            f_decoder_list.append(f_decoder_i)  # 将特征放到特征列表中，一共有4层，也就是列表中有4个元素，每个元素的维度减小，（512）--256--128--32---8
        # ###########################Decoder############################

        # 全连接层1/2/3，中间有个dropout，最后一层的输出是类型个数，不使用激活函数
        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        # keep_prob 每个元素被保留下来的概率，设置神经元被选中的概率,
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])  # 压缩维度，axis可以用来指定要删掉的为1的维度，此处要注意指定必须确保其是1，否则会报错 # B*N*1*C
        return f_out  # B*N*CLASS

    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch),
                self.Log_file)  # self.training_epoch = 0，保存路径输出到Log_file中
        self.sess.run(dataset.train_init_op)  # 将分好批次的数据集初始化第一次输入,处于while循环外，所以里面还有一行这个代码
        while self.training_epoch < self.config.max_epoch:  # 循环，直到达到设定的最大epoch
            t_start = time.time()  # 记录运行起始时间
            try:
                ops = [self.train_op,  # 训练损失函数达到最校的优化器
                       self.extra_update_ops,  # 更新后的变量
                       self.merged,  # 将所有summary全部保存到磁盘
                       self.loss,  # 损失函数数值
                       self.logits,  # B*N*C
                       self.labels,  # 标签值
                       self.accuracy]  # 精确度
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops,
                                                                         {self.is_training: True})  # 执行op，这里才传递出具体数值
                self.train_writer.add_summary(summary, self.training_step)  # 把每一步的结果添加进去
                t_end = time.time()  # 记录运行结束时间
                if self.training_step % 50 == 0:  # 如果是50的倍数
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)),
                            self.Log_file)  # 写入日志文件
                self.training_step += 1  # 会计算到501，但是ops中只有500个，超出维度
                # 由于一直在while中，下一个epoch的步数还会继续累加，直到while循环结束

            # 执行OutOfRangeError的原因是training_step超出维度，穷举了所有的数据batch，没有就超出维度了，这里的设计方便了不同数据集的设定
            # 比如，kitti数据集过于庞大，所以一个epoch会有3000+steps，要是在main函数中全部使用，数据量太大，分好batch，直接穷举就方便一些
            except tf.errors.OutOfRangeError:  # 也就是循环完成一个epoch，才计算下面的参数

                m_iou = self.evaluate(dataset)  # 平均交并比

                # 每次保存的都是比上一个的交并比好，所以100个epoch结束后，在mIou_list中找到最新的就是最好的
                if m_iou > np.max(self.mIou_list):  # 判断新计算的平均交并比是不是比以前的平均交并比大，所以第一个epoch一定会保存进snap
                    # Save the best model 保存最好的模型
                    snapshot_directory = join(self.saving_path, 'snapshots')  # 在result的log文件中创建快照
                    makedirs(snapshot_directory) if not exists(
                        snapshot_directory) else None  # 判断路径是否存在，不存在就创建路径os.makedirs
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)  # 保存
                    # 这个save了很多参数，学习率、准确度、第几步。模型各层的所有参数
                self.mIou_list.append(m_iou)  # 把每次计算的交并比都添加进去
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)  # 输出最好的平均交并比

                # 因为执行了try超出了范围，然后就会进入except中执行，在这里添加下一次的数据，继续在while循环中进行，直到training_epoch满足条件
                self.training_epoch += 1  # 训练epoch加一
                self.sess.run(dataset.train_init_op)  # 继续迭代输入数据流，多个批次数据集是并行计算的，计算一次为一个epoch
                # Update learning rate 更新学习率
                # tf.assign更新变量 self.learning_rate是一个变量
                # tf.multiply乘法计算
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                # 学习率是每次都衰减的，这也是一种优化方法，相当于梯度下降的步长减少，容易找到最小loss
                self.sess.run(op)  # 更新一个变量要使用sess.run初始化
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)  # 输出下一个EPOCH

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()  # 使用close关闭数据流

    def evaluate(self, dataset):

        # Initialise iterator with validation data 用验证数据初始化迭代器
        self.sess.run(dataset.val_init_op)  # 运行验证集数据，run相当于数据开关，这里就是验证集数据打开了，下面的就是用的这个数据
        # 这里表明是用验证集进行验证
        # 这句话只有一句，而不像 self.sess.run(dataset.train_init_op)有两句，是因为这个在上面也嵌套在循环中，一个epoch计算完后进入outrange，然后就会进入这里

        gt_classes = [0 for _ in range(self.config.num_classes)]  # 获取地表真实类别
        positive_classes = [0 for _ in range(self.config.num_classes)]  # 所有划分为正确类别
        true_positive_classes = [0 for _ in range(self.config.num_classes)]  # 正确的划分为正确类别
        val_total_correct = 0  # 验证集总体有效精度
        val_total_seen = 0  # 验证集总体有效点数
        for step_id in range(
                self.config.val_steps):  # val_steps = 100 因为验证集的数据很少，100*(20*40960) 100是steps，20是batch_size ，40960是points
            if step_id % 50 == 0:  # 如果是50的倍数
                print(str(step_id) + ' / ' + str(self.config.val_steps))  # 0/100 0/任何数都为0  50/100  没有100，因为range从0开始
                # 这句话是直接在终端打印出来的，不在log中保存
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)  # 运算参数
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})  # 不训练，参数不会更新
                # 也就是训练集得到的参数，验证集数据使用这些参数计算出归一化prob_logits和准确度,标签是验证集的标签
                pred = np.argmax(stacked_prob, 1)  # 选取最大的概率模型  b*n*1  如果做单分类问题，那么输出的值就取top1(最大，argmax)
                # 这句话就是将归一化特征值选取最大概率，返回的就当作预测标签

                # 忽略无效标签
                if not self.config.ignored_label_inds:  # 如果不是待忽略的标签值
                    pred_valid = pred  # 赋值给有效预测
                    labels_valid = labels  # 标签就是有效标签 （b*n）一维
                else:
                    # where 返回condition中值为True的位置的Tensor
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]  #
                    labels_valid = np.delete(labels, invalid_idx)  # 从labels中删除无效标签
                    labels_valid = labels_valid - 1  # 数值减一
                    pred_valid = np.delete(pred, invalid_idx)  # 同样删除预测中的无效标签

                # 计算指标
                correct = np.sum(pred_valid == labels_valid)  # 计算标签对应的个数
                val_total_correct += correct  # 所有正确的叫做总体正确个数
                val_total_seen += len(labels_valid)  # 记录有效的点数的个数

                conf_matrix = confusion_matrix(labels_valid, pred_valid,
                                               labels = np.arange(0, self.config.num_classes, 1))  # 计算混淆矩阵
                gt_classes += np.sum(conf_matrix, axis=1)  # 列维相加得到地表分类数
                positive_classes += np.sum(conf_matrix, axis=0)  # 行维相加得到预测分类数
                true_positive_classes += np.diagonal(conf_matrix)  # 对角线是正确分类的类别

            except tf.errors.OutOfRangeError:  # 超出范围停止
                break

        # 计算交并比
        iou_list = []  # 交并比列表
        for n in range(0, self.config.num_classes, 1):  # 0-19、
            # 真实预测/(地表+预测-真实预测)，这里-真实预测的原因是：这部分重叠了
            iou = true_positive_classes[n] / float(
                gt_classes[n] + positive_classes[n] - true_positive_classes[n])  # 计算交并比
            iou_list.append(np.nan_to_num(iou))
        mean_iou = sum(iou_list) / float(self.config.num_classes)  # 平均交并比，就是每一类的交并比求平均

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)  # 平均正确率
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)  # 百分比表示，保留小数点后一位
        s = '{:5.2f} | '.format(mean_iou)  # 平均交并比+|一竖
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)  # 输出一样长的-
        log_out(s, self.Log_file)  # 循环输出，每一类的交并比
        log_out('-' * len(s) + '\n', self.Log_file)  # 输出一样长的-，回车换行
        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits,
                                                            label_smoothing=0.1)
        # unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)

        if self.loss_type == 'lovas':
            logits = tf.reshape(logits, [-1, self.config.num_classes])  # [-1, n_class]
            probs = tf.nn.softmax(logits, axis=-1)  # [-1, class]
            labels = tf.reshape(labels, [-1])
            lovas_loss = L.lovasz_softmax(probs, labels, 'present')
            # output_loss = output_loss + lovas_loss

        return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):

        # 首先使用一个稀疏mlp，输出特征通道数减少一半 B*H*L*C,卷积核尺寸，[1, 1]，步长也是[1, 1]，True表示进行bn（批次归一化）
        # name + 'mlp1'是在tf_util中定义的tf.name_scope()、tf.variable_scope()会在模型中开辟各自的空间，而其中的变量均在这个空间内进行管理
        # 但是之所以有两个，主要还是有着各自的区别
        # 作用主要是集成化为模块，这样每个模块之间的变量互不影响，因为深度学习模型的变量非常多，不加上命名空间加以分组整理，将会成为可怕的灾难
        # d_out = [16, 64, 128, 256]
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        # 接着输入到核心模块中
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        # 经过构建模型，再次使用稀疏mlp，这次的输出是2倍,不使用激活函数
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        # 跳跃连接也不使用激活函数
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)  # 在求和后才使用了激活函数

    # 核卷积惨差块
    def residual_block(self, feature, xyz, neigh_idx, d_out, name, density_parameter, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        # [batch_size,核点数]
        #
        w = weight_variable([self.config.batch_size, self.config.num_kernels, f_pc.get_shape()[-1].value, d_out])
        f_pc = self.kernel_conv_plus(xyz, f_pc, neigh_idx, d_out, w, name + 'NCV', density_parameter=density_parameter,
                                     is_training=is_training)

        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        # d_in = d_out // 2
        d_in = feature.get_shape()[
            -1].value  # 获取特征维度的最后一维，feature.get_shape()，2d卷积后得到的feature是一个tensor，张量可以调用get_shape()获取形状，value调用具体的数值
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # 相关特征编码结构 B*N*K*10
        # 这里的mlp是对相关特征编码使用的mlp，有两个作用：1.特征非线性化，使得特征类型多样 2.将维度统一起来，感觉2更重要一些
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True,
                                      is_training)  # MLP操作 B*N*K*d_in

        # 上面是一条路，下面是另一条路
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2),
                                             neigh_idx)  # feature是4维，B*N*1*d_in压缩一维--》B*N*d_in，输出B*N*K*d_in（8）
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)  # 最后一维拼接B*N*K*2d_in=B*N*K*d_out（16）
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1',
                                    is_training)  # 注意力池化模块 B*N*1*（d_out /2）
        # 这里用d_out // 2的原因是为了使得d_in = d_out // 2，这样进行第二遍输入时，方便一些

        # 进行相同的第二遍操作
        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2),
                                             neigh_idx)  # f_pc_agg是4维，B*N*1*（d_out /2）压缩一维--》B*N*（d_out /2），输出B*N*K*（d_out /2）
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)  # 最后一维拼接B*N*K*d_out
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)  # 注意力池化模块 B*N*1*d_out
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):  # xyz B*N*3
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # 返回带有查询到的邻近点坐标的特征 B*N*K*3

        # tile 用来对张量(Tensor)进行扩展的，其特点是对当前张量内的数据进行一定规则的复制。最终的输出张量维度不变。
        # 相当于对应维度的内容扩张，但是整体的维度保持不变 B*N*1*3---1*1*k*1--》B*1 N*1 1*K 3*1---=B*N*K*3
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])  # 扩张维度xyz B*N*1*3  1*1*k*1
        relative_xyz = xyz_tile - neighbor_xyz  # B*N*K*3
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1,
                                             keepdims=True))  # B*N*K*1,对于最后一维进行平方、求和、再相加，计算后最后一维变成1，保持维度不变
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz],
                                     axis=-1)  # B*N*K*(3+3+3+1),在最后一维上拼接
        return relative_feature  # B*N*K*10

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)  # 圧缩维度，B*N*1*d---》B*N*d
        num_neigh = tf.shape(pool_idx)[-1]  # 获取临近点的数量 k
        d = feature.get_shape()[-1]  # 获取特征最后一维的维度，特征
        batch_size = tf.shape(pool_idx)[0]  # 获取特征的第一个维度，批次
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])  # B*N1*k ---> B*(N1*k)
        # tf.batch_gather 按照 pool_idx的索引在feature中个搜索点云
        # 注意，这里feature是三维，pool_idx是二维，由于要按照pool_idx的索引在feature中选取，所以提取的是（N1*k）个d，就是N1*k个d向量
        pool_features = tf.batch_gather(feature, pool_idx)  # B*（N1*k）*d 三维
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])  # B*N1*k*d 四维
        # 下采样选取的是k个邻近点最大特征的值，相当于池化操作
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)  # B*N1*1*d  输入与输出的维度不变
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        # 最邻近插值（上采样）所以找的是最近的点 [B, up_num_points, 1]，也就是第三维是1的原因
        feature = tf.squeeze(feature, axis=2)  # 和下采样一样，圧缩维度，B*N1*1*d---》B*N1*d
        batch_size = tf.shape(interp_idx)[0]  # 获取特征的第一个维度，批次，B*N*1
        up_num_points = tf.shape(interp_idx)[1]  # 获取特征的第二个维度，上采样点的个数，B*N*1
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])  # B*N*1-->B*(N*1)
        # 注意，这里feature是三维，pool_idx是二维，由于要按照interp_idx的索引在feature中选取，所以提取的是（N*1）个d，就是N*1个d向量
        # 上采样中，interp_idx的个数大于feature的个数，也就是N>N1，所以会出现重复读取feature，也就是实现了上采样，相当于有些点赋予了相同特征
        interpolated_features = tf.batch_gather(feature, interp_idx)  # B*(N1*1)*d 三维
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)  # 扩张维度 B*N1*1*d 使得输入与输出的维度不变
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc B*N*3  neighbor_idx B*N*K
        # gather the coordinates or features of neighboring points 收集相邻点的坐标或特征
        batch_size = tf.shape(pc)[0]  # 获取点云的第一个维度，也就是批次大小 B*N*F
        num_points = tf.shape(pc)[1]  # 获取点云的第二个维度，也就是点云的个数 B*N*F
        d = pc.get_shape()[2].value  # 获取点云的第二个维度，也就是坐标具体的数值
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])  # 修改邻近点索引的尺寸，二维数组 B*（N*K）
        features = tf.batch_gather(pc, index_input)  # tf.batch_gather要保证在batch上维度的统一，使用index_input查询pc中相关的点
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])  # 修改特征为B*N*K*d
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        # 这里写的是d_out，在模型中这个是d_out//2，也就是实际运算中是一半
        batch_size = tf.shape(feature_set)[0]  # 获取特征的第一个维度，也就是批次大小 B*N*k*d_out
        num_points = tf.shape(feature_set)[1]  # 获取特征的第二个维度，也就是点云的个数 B*N*k*d_out
        num_neigh = tf.shape(feature_set)[2]  # 获取特征的第三个维度，也就是邻近点的数量  B*N*k*d_out
        d = feature_set.get_shape()[3].value  # 获取特征的第四个维度，也就是坐标具体的数值  B*N*k*d_out(16)
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])  # （b*n）*k*d_out，为了和注意力机制模型相乘计算

        # tf.layers.dense改变f_reshaped的最后一维，d---》d，这里没有变化
        # 这两句就是用来构建注意力机制模型的
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False,
                                         name=name + 'fc')  # 全连接层 （b*n）*k*d
        att_scores = tf.nn.softmax(att_activation, axis=1)  # softmax函数之前要用全连接 （b*n）*k*d

        f_agg = f_reshaped * att_scores  # 相乘，得到注意力特征 （b*n）*k*d
        f_agg = tf.reduce_sum(f_agg, axis=1)  # 求和 （b*n）*d
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])  # B*N*1*d
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)  # mlp

        return f_agg

    # 刚性卷积核 卷积核不变
    def kernel_conv(self, xyz, features, neigh_idx, d_out, K_values, name, density_parameter, is_training,
                    modulated=False):
        points_dim = xyz.get_shape()[-1].value
        d_in = xyz.get_shape()[-1].value
        # points_dim = 3
        KP_extent = 1.0
        num_kpoints = int(K_values.shape[1])
        if modulated:
            offset_dim = (points_dim + 1) * (num_kpoints - 1)
        else:
            offset_dim = points_dim * (num_kpoints - 1)
        shape0 = K_values.shape.as_list()  # [1,15,8,16]
        w0 = tf.Variable(
            tf.zeros([self.config.batch_size, shape0[1], offset_dim], dtype=tf.float32),
            name='offset_mlp_weights')

        b0 = tf.Variable(tf.zeros([offset_dim], dtype=tf.float32), name='offset_mlp_bias')
        print(w0, '11111111111111111', features, '2222222222222222222')
        features0 = unary_convolution(features, w0) + b0
        print(features0)
        # features0 = features
        if modulated:

            # Get offset (in normalized scale) from features
            offsets = features0[:, :points_dim * (num_kpoints - 1)]
            offsets = tf.reshape(offsets, [-1, (num_kpoints - 1), points_dim])

            # Get modulations
            modulations = 2 * tf.sigmoid(features0[:, points_dim * (num_kpoints - 1):])

            #  No offset for the first Kernel points
            offsets = tf.concat([tf.zeros_like(offsets[:, :1, :]), offsets], axis=1)
            modulations = tf.concat([tf.zeros_like(modulations[:, :1]), modulations], axis=1)

        else:

            # Get offset (in normalized scale) from features
            offsets = tf.reshape(features0, [-1, -1, (num_kpoints - 1), points_dim])

            #  No offset for the first Kernel points
            offsets = tf.concat([tf.zeros_like(offsets[:, :, :1, :]), offsets], axis=2)

            # No modulations
            modulations = None

        # Rescale offset for this layer
        offsets *= KP_extent

        K_radius = 1.5 * KP_extent / density_parameter

        K_points_numpy = kernels.load_kernels(K_radius,
                                              num_kpoints,
                                              num_kernels=1,
                                              dimension=points_dim,
                                              fixed='center')
        K_points_numpy = K_points_numpy.reshape((num_kpoints, points_dim))
        K_points = tf.Variable(K_points_numpy.astype(np.float32),
                               name='kernel_points',
                               trainable=False,
                               dtype=tf.float32)

        n_kp = int(K_points.shape[0])
        neighbors = self.gather_neighbour(xyz, neigh_idx)  # B, N, k, 3

        neighbors = neighbors - tf.expand_dims(xyz, 2)  # B,N,K,3
        deformed_K_points = tf.add(offsets, K_points, name='deformed_KP')

        neighbors = tf.expand_dims(neighbors, 3)
        neighbors = tf.tile(neighbors, [1, 1, 1, n_kp, 1])

        differences = neighbors - deformed_K_points  # [b,n,k,15,3]

        # differences = neighbors -tf.expand_dims(deformed_K_points, 2)
        # print(differences,'dierge')
        sq_distances = tf.reduce_sum(tf.square(differences), axis=4)

        all_weights = tf.maximum(1 - tf.sqrt(sq_distances) / KP_extent, 0.0)  # [(?, ?, ?, 15)]

        all_weights = tf.transpose(all_weights, [0, 1, 3, 2])
        neighbors_1nn = tf.argmin(sq_distances, axis=3, output_type=tf.int32)
        all_weights *= tf.one_hot(neighbors_1nn, n_kp, axis=2, dtype=tf.float32)
        # features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)

        # [b,n,k,d_in]
        neighborhood_features = self.gather_neighbour(tf.squeeze(features, axis=2), neigh_idx)

        # 核点相乘
        weighted_features = tf.matmul(all_weights, neighborhood_features)

        weighted_features = tf.transpose(weighted_features, [0, 2, 1, 3])
        # 权重矩阵相乘
        kernel_outputs = tf.matmul(weighted_features, K_values)

        # Convolution sum to get [n_points, out_fdim]
        output_features = tf.reduce_sum(kernel_outputs, axis=1)

        output_features = tf.nn.leaky_relu(output_features)
        output_features = tf.expand_dims(output_features, axis=2)
        print(output_features)
        return output_features

    def kernel_conv_plus(self, xyz, features, neigh_idx, d_out, K_values, name, density_parameter, is_training,
                         modulated=True):
        points_dim = xyz.get_shape()[-1].value
        d_in = xyz.get_shape()[-1].value
        # points_dim = 3
        KP_extent = 1.0
        num_kpoints = int(K_values.shape[1])
        if modulated:
            offset_dim = (d_in + 1) * (num_kpoints - 1)
        else:
            offset_dim = points_dim * (num_kpoints - 1)
        shape0 = K_values.shape.as_list()  # [3,16,8,16]
        # w0 = tf.Variable(
        #     tf.zeros([self.config.batch_size, shape0[2], offset_dim], dtype=tf.float32),
        #     name='offset_mlp_weights')
        #
        # b0 = tf.Variable(tf.zeros([offset_dim], dtype=tf.float32), name='offset_mlp_bias')
        # featuresoff = tf.squeeze(features)
        # features0 = unary_convolution(featuresoff, w0) + b0

        # original
        features = helper_tf_util.conv2d(features, features.get_shape()[-1].value, [1, 1], name + 'mlp1', [1, 1],
                                         'VALID', True, is_training)
        neigh_feat = self.gather_neighbour(tf.squeeze(features, axis=2), neigh_idx)
        neigh_xyz = self.gather_neighbour(xyz, neigh_idx)
        tile_feat = tf.tile(features, [1, 1, self.config.k_n, 1])  # B, N, k, d_out/2
        tile_xyz = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, self.config.k_n, 1])
        feat_info = tf.concat([neigh_feat - tile_feat, tile_feat], axis=-1)
        neigh_xyz_offsets = helper_tf_util.conv2d(feat_info, xyz.get_shape()[-1].value, [1, 1], name + 'mlp5', [1, 1],
                                                  'VALID', True, is_training)  # B, N, k, 3
        shifted_neigh_xyz = neigh_xyz + neigh_xyz_offsets
        xyz_info = tf.concat([neigh_xyz - tile_xyz, shifted_neigh_xyz, tile_xyz], axis=-1)  # B, N, k, 9
        neigh_feat_offsets = helper_tf_util.conv2d(xyz_info, features.get_shape()[-1].value, [1, 1], name + 'mlp6',
                                                   [1, 1], 'VALID', True, is_training)  # B, N, k, d_out
        shifted_neigh_feat = neigh_feat + neigh_feat_offsets

        feat_info = tf.concat([shifted_neigh_feat, feat_info], axis=-1)
        feat_encoding = helper_tf_util.conv2d(feat_info, features.get_shape()[-1].value, [1, 1], name + 'mlp8', [1, 1],
                                              'VALID', True,
                                              is_training)

        # original

        # new
        # features = helper_tf_util.conv2d(features, features.get_shape()[-1].value, [1, 1], name + 'mlp1', [1, 1],
        #                                  'VALID', True, is_training)
        # neigh_feat = self.gather_neighbour(tf.squeeze(features, axis=2), neigh_idx)
        # neigh_xyz = self.gather_neighbour(xyz, neigh_idx)
        # tile_feat = tf.tile(features, [1, 1, self.config.k_n, 1])  # B, N, k, d_out/2
        # tile_xyz = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, self.config.k_n, 1])
        # xyz_info = tf.concat([neigh_xyz - tile_xyz, tile_xyz], axis=-1)
        # neigh_feat_offsets = helper_tf_util.conv2d(xyz_info, features.get_shape()[-1].value, [1, 1], name + 'mlp6',
        #                                            [1, 1], 'VALID', True, is_training)
        # shifted_neigh_feat = neigh_feat + neigh_feat_offsets
        # feat_info = tf.concat([neigh_feat - tile_feat, shifted_neigh_feat, tile_feat], axis=-1)
        # feat_encoding = helper_tf_util.conv2d(feat_info, features.get_shape()[-1].value, [1, 1], name + 'mlp8', [1, 1],
        #                                       'VALID', True,
        #                                       is_training)

        # new

        # if modulated:
        #     # offsets[n_points, n_kpoints, dim]
        #     # Get offset (in normalized scale) from features
        #     # (3, 16, 1, 45)
        #
        #     offsets = features0[:self.config.batch_size, :, :points_dim * (num_kpoints - 1)]
        #
        #     offsets = tf.reshape(offsets, [self.config.batch_size, -1, (num_kpoints - 1), points_dim])
        #
        #     # Get modulations
        #     modulations = 2 * tf.sigmoid(features0[:, :, points_dim * (num_kpoints - 1):])
        #
        #     offsets = tf.concat([tf.zeros_like(offsets[:, :, :1, :]), offsets], axis=2)
        #
        #     modulations = tf.concat([tf.zeros_like(modulations[:, :, :1]), modulations], axis=-1)
        # else:
        #
        #     # Get offset (in normalized scale) from features
        #     offsets = tf.reshape(features0, [self.config.batch_size, -1, (num_kpoints - 1), points_dim])
        #
        #     #  No offset for the first Kernel points
        #     offsets = tf.concat([tf.zeros_like(offsets[:, :, :1, :]), offsets], axis=2)
        #
        #     # No modulations
        #     modulations = None

        # Rescale offset for this layer
        # offsets *= KP_extent

        K_radius = 1.5 * KP_extent / density_parameter

        K_points_numpy = kernels.load_kernels(K_radius,
                                              num_kpoints,
                                              num_kernels=1,
                                              dimension=points_dim,
                                              fixed='center')
        K_points_numpy = K_points_numpy.reshape((num_kpoints, points_dim))
        K_points = tf.Variable(K_points_numpy.astype(np.float32),
                               name='kernel_points',
                               trainable=False,
                               dtype=tf.float32)

        n_kp = int(K_points.shape[0])
        neighbors = self.gather_neighbour(xyz, neigh_idx)  # B, N, k, 3

        neighbors = neighbors - tf.expand_dims(xyz, 2)  # B,N,K,3
        # (3, 8, 16, 3)
        neighbors = tf.layers.batch_normalization(neighbors,
                                                  momentum=0.99,
                                                  epsilon=1e-6,
                                                  training=is_training)

        # deformed_K_points = tf.add(offsets, K_points, name='deformed_KP')

        neighbors = tf.expand_dims(neighbors, 3)
        neighbors = tf.tile(neighbors, [1, 1, 1, n_kp, 1])

        differences = neighbors - K_points  # [b,n,k,15,3]

        # differences = neighbors - tf.expand_dims(deformed_K_points, 2)

        sq_distances = tf.reduce_sum(tf.square(differences), axis=4)

        all_weights = tf.maximum(1 - tf.sqrt(sq_distances) / KP_extent, 0.0)  # [(?, ?, ?, 15)]

        all_weights = tf.transpose(all_weights, [0, 1, 3, 2])
        neighbors_1nn = tf.argmin(sq_distances, axis=3, output_type=tf.int32)
        all_weights *= tf.one_hot(neighbors_1nn, n_kp, axis=2, dtype=tf.float32)

        # features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)

        # [b,n,k,d_in]
        neighborhood_features = feat_encoding

        # neighborhood_features = self.gather_neighbour(tf.squeeze(features, axis=2), neigh_idx)

        # 核点相乘
        weighted_features = tf.matmul(all_weights, neighborhood_features)

        # if modulations is not None:
        #     weighted_features *= tf.expand_dims(modulations, -1)

        weighted_features = tf.transpose(weighted_features, [0, 2, 1, 3])
        # 权重矩阵相乘
        kernel_outputs = tf.matmul(weighted_features, K_values)

        # Convolution sum to get [n_points, out_fdim]
        output_features = tf.reduce_sum(kernel_outputs, axis=1)

        output_features = tf.nn.leaky_relu(output_features)
        output_features = tf.expand_dims(output_features, axis=2)

        return output_features

    @staticmethod
    def unary_convolution(features,
                          K_values):

        """
        Simple unary convolution in tensorflow. Equivalent to matrix multiplication (space projection) for each features
        :param features: float32[n_points, in_fdim] - input features
        :param K_values: float32[in_fdim, out_fdim] - weights of the kernel
        :return: output_features float32[n_points, out_fdim]
        """

        return tf.matmul(features, K_values)
