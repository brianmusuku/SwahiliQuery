import tensorflow as tf
import numpy as np
import csv, re, random
import dataProcess


X_train,X_lengths, Y_train, dimen, n_steps, X_test, X_test_lengths, y_test, similarityData, memory, memorySl = dataProcess.main()


def _last_relevant(output, length):
    """
    get the last relevant tensor in an rnn
    output given real tensor length
    """
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (tf.cast(length, tf.int32) - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant

def attention(Y, dimen):
    # [batch_size x seq_len x dim]  -- hidden states
    #Y = tf.constant(np.random.randn(batch_size, seq_len, dim), tf.float32)
    # [batch_size x dim]            -- h_N
    #h = tf.constant(np.random.randn(batch_size, dim), tf.float32)
    initializer = tf.random_uniform_initializer()
    W = tf.get_variable("weights_Y", [dimen, dimen], initializer=initializer)
    w = tf.get_variable("weights_w", [dimen], initializer=initializer)

    # [batch_size x seq_len x dim]  -- tanh(W^{Y}Y)
    M = tf.tanh(tf.einsum("aij,jk->aik", Y, W))
    # [batch_size x seq_len]        -- softmax(Y w^T)
    a = tf.nn.softmax(tf.einsum("aij,j->ai", M, w))
    # [batch_size x dim]            -- Ya^T
    r = tf.einsum("aij,ai->aj", Y, a)
    return r, a


def mem_attention(mem, question, dimen):
    """
    mem == [memSlots x dimen]
    question == [numQuestions x dimen]
    """
    initializer = tf.random_uniform_initializer()
    W = tf.get_variable("w10", [16, 16], initializer=initializer)
    #do a dot prod multiplication
    #[numQuestions x dimen] , [dimen, memslots] == [numQuestions, memSlots]
    similarity = tf.matmul(question, tf.transpose(mem)) 
    a = similarity
    #for each similarity get the mem
    # [numQuestions, memSlots] [memSlots x dimen]
    r = tf.matmul(a, W)
    return r, a


classes = 16
target = tf.placeholder(tf.float32, [None, classes],name ="y_true")
data = tf.placeholder(tf.float32, [None, n_steps,dimen], name ="x_true") #Number of examples, number of input, dimension of each input
featureExist = tf.placeholder(tf.float32, [None, classes], name ="x_feature")
sl = tf.placeholder(tf.float32, [None])
num_hidden = 16
learning_rate =1e-3
cell = tf.contrib.rnn.GRUCell(num_hidden)
cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=.70,  output_keep_prob=.70)
outputs, last_states = tf.nn.dynamic_rnn(cell = cell, dtype = tf.float32, sequence_length = sl, inputs = data)
last, att = attention(outputs, num_hidden)
#last = _last_relevant(outputs, sl)

with tf.variable_scope('attention_', reuse=None) as f:
    mem = tf.placeholder(tf.float32, [None, n_steps,dimen], name ="mem_contents")
    mem_sl = tf.placeholder(tf.float32, [None])
    mem_outputs, mem_states = tf.nn.dynamic_rnn(cell = cell, dtype = tf.float32, sequence_length = mem_sl, inputs = mem)
    #mem_last = _last_relevant(mem_outputs, mem_sl)
    mem_last, _ = attention(mem_outputs, num_hidden)
mem_last, mem_att = mem_attention(mem_last, last, num_hidden)
laste = tf.concat([mem_last, featureExist], axis=1)
W2 = tf.get_variable("M",shape=[classes * 2, classes],initializer=tf.truncated_normal_initializer(stddev=0.1))
W1 = tf.get_variable("M1",shape=[classes, classes],initializer=tf.truncated_normal_initializer(stddev=0.1))
wes = tf.matmul(laste, W2)
wes = tf.matmul(wes, W1)

cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=wes, labels=target))

correct_prediction = tf.equal(tf.argmax(wes, 1), tf.argmax(target, 1))
# Calculate accuracy for  test set
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
saver = tf.train.Saver()

init_op = tf.global_variables_initializer()
sess = tf.Session()
summary_writer = tf.summary.FileWriter("/tmp/charLevel")
summary_writer.add_graph(sess.graph)
sess.run(init_op)
batch_size = int(len(X_train))
no_of_batches = int(len(X_train) / batch_size)
epoch = 700
modelName = 'inflationModel2'
try:
    saver = tf.train.import_meta_graph(modelName+".meta")
    print("Loading variables from '%s'." % modelName)
    saver.restore(sess, tf.train.latest_checkpoint('./trainedModel2'))
    print("Success: Loaded variables from '%s'." % modelName)
except IOError:
    print("Not found: Creating new '%s'." % modelName)
testDict = {data: X_test, target: y_test, sl:X_test_lengths, mem:memory, mem_sl:memorySl, featureExist:similarityData[0]}
trainDict = {data: X_train, target: Y_train, sl:X_lengths, mem:memory, mem_sl:memorySl, featureExist:similarityData[1]}
#training & testing modes
mode = 0
if mode == 0:
    for i in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
            inp, out, leno, feat = X_train[ptr:ptr+batch_size], Y_train[ptr:ptr+batch_size], X_lengths[ptr:ptr+batch_size], similarityData[1][ptr:ptr+batch_size]
            ptr+=batch_size
            sess.run(train_op, {data: inp, target: out, sl:leno, mem:memory, mem_sl:memorySl, featureExist: feat})
        costsss =float(sess.run(cost, trainDict))
        testAcc = sess.run(accuracy, testDict)
        trainAcc = sess.run(accuracy, trainDict)
        test_cost = sess.run(cost, testDict)
        print ("Epoch "+str(i)+" Train set cost: "+str(round(costsss, 2)),"\t Test cost: ", round(test_cost, 2),"\ttestAccuracy:", testAcc, end=" " )
        print("\tTrain Accuracy: "+str(round(trainAcc, 2)))
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=costsss, tag="training_cost")
        episode_summary.value.add(simple_value=test_cost, tag="test_cost")
        episode_summary.value.add(simple_value=testAcc, tag="test_accuracy")
        summary_writer.add_summary(episode_summary, i)
        summary_writer.flush()
        if testAcc >60.54:
            saver.save(sess, modelName)
            #sys.exit();