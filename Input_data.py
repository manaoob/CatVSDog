import tensorflow as tf
import numpy as np
import os

img_width = 200
img_height = 208

"""
    Args:
        file_dir: file dircetory
    return
        list of images and labels
"""
def get_files(file_dir):
    cats=[]
    label_cats=[]
    dogs=[]
    label_dogs=[]
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats \nThere are %d dogs'%(len(cats), len(dogs)))

    # hstack()-->参数tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组
    """
    example
    其实就是水平(按列顺序)把数组给堆叠起来，vstack()函数正好和它相反
    import numpy as np
        a=[1,2,3]
        b=[4,5,6]
        print(np.hstack((a,b)))

        输出：[1 2 3 4 5 6 ]
    """
    image_list = np.hstack((cats,dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list,label_list])
    temp = temp.transpose()  # 转置~~
    np.random.shuffle(temp)  # 数组的打乱操作 即将temp打乱

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    Args:
    :param image: list type
    :param label: list type
    :param image_W: image width
    :param image_H: image height
    :param batch_size: batch size
    :param capacity: the maximum element in queue
    :return:
        image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
        label_batch: 1D tensor [batch_size] dtype-tf.int32
    """

    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)

    # 生成一个输入队列（具体用法见有道云笔记）
    input_queue = tf.train.slice_input_producer([image,label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    ###########################
    # 此处还应该有“特征工程” --> 提高模型的精度
    ###########################
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size = batch_size,
                                              num_threads = 64,
                                              capacity = capacity)

    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch



