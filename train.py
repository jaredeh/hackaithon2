#! /usr/bin/env python3
from tinygrad import Tensor, nn
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.engine.jit import TinyJit
import datetime
import numpy as np
from tqdm import trange
from time import sleep
import argparse
import gc
import os
import random
import polars as pl

from model import SimpleFNN
"""
This trains the model based on a parquet dataset using the tinygrad framework.
"""


def get_timestamp(timestamp=None):
    if timestamp is not None:
        return timestamp
    return datetime.datetime.utcnow()

def get_runtime(starttime, endtime=None):
    if endtime is None:
        return get_timestamp() - starttime
    return endtime - starttime


def evaluate(model, X_test, Y_test, BS=1, return_predict=False):
  Tensor.training = False
  print(Y_test.shape)
  def numpy_eval(Y_test):
    Y_test_preds_out = np.zeros(list(Y_test.shape))
    for i in trange((len(Y_test)-1)//BS+1):
      x = Tensor(X_test[i:(i+1)])
      out = model.forward(x) if hasattr(model, 'forward') else model(x)
      Y_test_preds_out[i:(i+1)] = out.numpy()[0]
    return (Y_test_preds_out - Y_test).mean(), Y_test_preds_out

  acc, Y_test_pred = numpy_eval(Y_test)
  print("test set error is %f" % acc)
  return (acc, Y_test_pred) if return_predict else acc


def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=lambda out,y: out.sparse_categorical_crossentropy(y),
        noloss=False):

  def train_step(x, y):
    # network
    out = model.forward(x) if hasattr(model, 'forward') else model(x)
    # print(f"out: {out.realize().numpy()}")
    # print(f"y: {y.realize().numpy()}")
    loss = lossfn(out, y)
    optim.zero_grad()
    loss.backward()
    if noloss: del loss
    optim.step()
    if noloss: return (None, None)
    accuracy = (out - y).mean()
    return loss.realize(), accuracy.realize()

  train_step = TinyJit(train_step)

  with Tensor.train():
    losses, accuracies = [], []
    for i in (t := trange(steps)):
      samp = np.random.randint(0, X_train.shape[0], size=(BS))
      #print(f"samp: {samp}")
      x = Tensor(X_train[samp], requires_grad=False)
      y = Tensor(Y_train[samp])
      loss, accuracy = train_step(x, y)
      # printing
      if not noloss:
        loss, accuracy = loss.numpy(), accuracy.numpy()
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
  return [losses, accuracies]


class StorgeTrainer():

    def __init__(self, config, dry_run=False, verbose=False):

        self.epochs = config.get('epochs', 1)
        self.batch_size = config.get('batch_size', 30000)
        self.trainingset_ratio = config.get('trainingset_ratio', 0.8)
        self.verbose = verbose
        self.dry_run = dry_run
        self.max_length = config.get('maxlength',0)
        self.outputdir = config.get('outputdir','output')
        self.inputdir = config.get('inputdir','data')

    def load_samples(self):
        # find files int inputdir
        files = os.listdir(self.inputdir)
        print(files)
        return files

    def get_x_y(self, sample):
        fname = os.path.join(self.inputdir, sample)
        df = pl.read_parquet(fname)
        df_x = df.select(pl.all().exclude('time_idle'))
        df_y = df.select(pl.col('time_idle'))
        # convert to numpy arrays for tinygrad
        x = df_x.to_numpy()
        y = df_y.to_numpy()
        return x,y

    def test0_x_y(self, sample):
        fname = os.path.join(self.inputdir, sample)
        # read as polars dataframe
        df = pl.read_parquet(fname)
        #remove all but last_service == 7.0
        df =  df.filter(pl.col("last_service") != 7.0)
        print(df)
        df_x = df.select(pl.all().exclude('time_idle'))
        print(df_x)
        raise
        df_y = df.select(pl.col('time_idle'))
        x = df_x.to_numpy()
        y = df_y.to_numpy()
        return x,y

    def test1_x_y(self, sample):
        fname = os.path.join(self.inputdir, sample)
        # read as polars dataframe
        df = pl.read_parquet(fname)
        #remove all but last_service == 7.0
        df =  df.filter(pl.col("last_service") == 7.0)
        print(df)
        df_x = df.select(pl.all().exclude('time_idle'))
        df_y = df.select(pl.col('time_idle'))
        x = df_x.to_numpy()
        y = df_y.to_numpy()
        return x,y

    def run(self):
        starttime = get_timestamp()
        print()
        print("Beginning Training")
        # First we load information about the individual sample files
        samples = self.load_samples()
        print(f"Samples Loaded: {len(samples)}")
        # Now we sort out the counts of the training/evaluating sets
        setlength = len(samples)
        if (self.max_length > 0) and (self.max_length < setlength):
            sample_count = self.max_length
        else:
            sample_count = setlength
        trainingset_count = int(sample_count*self.trainingset_ratio)
        print(f"Sample Count: {sample_count}")
        print(f"Trainingset Count: {trainingset_count}")

        # We shuffle the samples and split them into training and evaluating sets
        shuffled_samples = random.sample(samples[0:setlength], sample_count)
        training_samples = shuffled_samples[0:trainingset_count]
        evaluating_samples = shuffled_samples[trainingset_count:sample_count]
        print(f"Training Samples: {len(training_samples)}")
        print(f"Evaluating Samples: {len(evaluating_samples)}")

        model = self.get_compiled_model(samples[0])
        self.fit(model, training_samples, starttime)

        self.evaluate(model, evaluating_samples, starttime)

        # Set up output dir
        os.makedirs(self.outputdir, exist_ok=True)

        basefilename = os.path.join(self.outputdir, f"{starttime.strftime('%Y%m%d%H%M%S')}")

        # Save the trained model if needed
        model.save(f"{basefilename}.safe")
        model = self.get_compiled_model(samples[0])
        self.evaluate(model, evaluating_samples, starttime)
        model.load(f"{basefilename}.safe")
        self.evaluate(model, evaluating_samples, starttime)

    def get_compiled_model(self, sample):
        x, y = self.get_x_y(sample)
        dataset_size = x.shape[1]
        output_size = y.shape[1]
        model = SimpleFNN(dataset_size, output_size)
        print(f"Model: x:{dataset_size} -> y:{output_size}")
        parameters = get_parameters(model)
        print("parameter count", len(parameters))
        for k in parameters:
            print(f"   {k}")
        # o = model.forward(Tensor(x))
        # print(o.realize().numpy())
        # print("model done")
        return model

    def fit(self, model, samples, starttime):
        setlength = len(samples)
        # model.to(device)
        parameters = get_parameters(model)
        #lossfn=lambda out,y: (y)
        lossfn=lambda out,y: out.binary_crossentropy_logits(y)
        lr = 5e-3
        for epoch in range(self.epochs):
            for i in range(setlength):
                sample = samples[i]
                x, y = self.get_x_y(sample)
                optimizer = optim.Adam(get_parameters(model), lr=lr)
                train(model, x, y, optimizer, len(x), BS=self.batch_size, lossfn=lossfn)
            lr /= 1.2

    def evaluate(self, model, samples, starttime):
        runtime = datetime.timedelta(0)

        setlength = len(samples)
        if (self.max_length > 0) and (self.max_length < setlength):
            setlength = self.max_length
        x,y = self.test1_x_y(samples[0])
        a = evaluate(model,x,y, return_predict=True, BS=1)
        #print(a)
        x,y = self.test0_x_y(samples[0])
        a = evaluate(model,x,y, return_predict=True, BS=1)
        #print(a)
        return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxlength', help='How many samples to work off of', type=int, default=0)
    parser.add_argument('--epochs', help='epochs for training', type=int, default=1)
    parser.add_argument('--batch_size', help='batch_size for training', type=int, default=30000)
    parser.add_argument('--trainingset_ratio', help='Ratio for trainingset vs evaluation / test set', type=float, default=0.8)
    parser.add_argument('--outputdir', help='Directory for saved files', type=str, default='output')
    parser.add_argument('--inputdir', help='Directory for input data files', type=str, default='data')
    parser.add_argument('--dryrun', help='run training and inspect model locally mode', action='store_true')
    parser.add_argument('--verbose', help='verbose output', action='store_true')
    args = parser.parse_args()

    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'trainingset_ratio': args.trainingset_ratio,
        'outputdir': args.outputdir,
        'inputdir': args.inputdir,
        'maxlength': args.maxlength,
    }
    t = StorgeTrainer(config, dry_run=args.dryrun, verbose=args.verbose)
    t.run()
