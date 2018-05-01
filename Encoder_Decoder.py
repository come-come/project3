import numpy as np
import tensorflow as tf
import helpers
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

flags = tf.flags

flags.DEFINE_string("save_path", None, "Model output directory")
FLAGS = flags.FLAGS

PAD = 0
EOS = 1

def main(_):
    vocab_size = 10
    input_embedding_size = 20

    encoder_hidden_units = 20
    decoder_hidden_units = 20

    with tf.name_scope('encoder_inputs'):
        encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    with tf.name_scope('decoder_targets'):
        decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

    with tf.name_scope('decoder_iputs'):
        decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

    with tf.name_scope('embeddings'):
        embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -0.1, 1.0), dtype=tf.float32)

    with tf.name_scope('encoder_inputs_embedded'):
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    with tf.name_scope('decoder_inputs_embedded'):
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

    with tf.name_scope('encoder_cell'):
        encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
    with tf.name_scope('encoder_dynamic'):
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, dtype=tf.float32, time_major=True,)
    del encoder_outputs

    encoder_final_state

    with tf.name_scope('decoder_cell'):
        decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
    with tf.name_scope('decoder_dynamic'):
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
            decoder_cell, decoder_inputs_embedded, initial_state=encoder_final_state,
            dtype=tf.float32, time_major=True, scope="plain_decoder",)

    with tf.name_scope('decoder_logits'):
        decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
    with tf.name_scope('decoder_prediction'):
        decoder_prediction = tf.argmax(decoder_logits, 2)
    decoder_logits

    with tf.name_scope('stepwise_cross_entropy'):
        stepwise_cross_entropy =  tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
            logits=decoder_logits,)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(stepwise_cross_entropy)
    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer().minimize(loss)

        sess.run(tf.global_variables_initializer())

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)

    with sv.managed_session() as session:
        batch_ = [[6], [3,4], [9,8,7]]
        batch_, batch_length_ = helpers.batch(batch_)
        #print('batch_encoded:\n' + str(batch_))

        din_, dlen_ = helpers.batch(np.ones(shape=(3,1), dtype=np.int32), max_sequence_length=4)

        #print('decoder inputs:\n' + str(din_))

        pred_ = sess.run(decoder_prediction, feed_dict={
            encoder_inputs:batch_,
            decoder_inputs: din_,
        })

        if FLAGS.save_path:
            sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

        #print('decoder predictions:\n' + str(pred_))

    batch_size = 100
    batches = helpers.random_sequences(length_from=3, length_to=8, vocab_lower=2, vocab_upper=10, batch_size=batch_size)

    print('head of the batch:')
#   for seq in next(batches)[:10]:
#       print(seq)

    def next_feed():
        batch = next(batches)
#       print('nex_feed batch EOS:{}'.format(EOS))
        for seq in batch:
            print(seq)
        encoder_inputs_, _ = helpers.batch(batch)
#       print('encode_input {}'.format(encoder_inputs_))
        decoder_targets_, _ = helpers.batch(
            [((sequence) + [EOS]) for sequence in batch]
        )
#       print('decoder_targets_{}'.format(decoder_targets_))
        decoder_inputs_, _ = helpers.batch(
            [ ([EOS] + (sequence)) for sequence in batch]
        )
#       print('decode_input {}'.format(decoder_inputs_))
        return {
            encoder_inputs: encoder_inputs_,
            decoder_inputs: decoder_inputs_,
            decoder_targets: decoder_targets_,
        }

    loss_track = []

    max_batches = 3001
    batches_in_epoch = 1000

    try:
        for batch in range(max_batches):
            fd = next_feed()
            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
#               print('batch{}'.format(batch))
#               print('minibatch loss: {}'.format(sess.run(loss, fd)))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                    print(' sample{}:'.format(i+1))
                    print('  input   > {}'.format(inp))
                    print('  predicted > {}'.format(pred))
                    if i >= 2:
                        break
    except KeyboardInterrupt:
        print('training interrupted')
    plt.plot(loss_track)
#   plt.show()
    print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))

if __name__ == "__main__":
    tf.app.run()