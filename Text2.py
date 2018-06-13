import os
import numpy as np
import tensorflow as tf
import Input_data
import Model

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE=16
CAPACITY = 2000
MAX_STEP = 15000
learning_rate = 0.0001


def run_training():
    train_dir =''
    logs_train_dir=''

    train, train_label = Input_data.get_files(train_dir)

    train_batch, train_label_batch = Input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    train_logits = Model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = Model.losses(train_logits, train_label_batch)
    train_op = Model.train(train_loss, learning_rate)
    train_accuracy = Model.evaluation(train_logits,train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss,
                                             train_accuracy])
            if step % 50 == 0:
                print('step %d, train loss = %.2f, train accuracy = %.2f%%'%(step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                # 保存模型
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(thread)
    sess.close()
