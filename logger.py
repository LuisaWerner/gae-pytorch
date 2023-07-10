import time
import numpy as np


class RunLogger(object):
    """
    This tracks the statistics per run. Eg. all training and validation accuracies per epoch, epoch times, losses as lists
    It tracks one final test accuracy and one final validation accuracy computed after training
    It tracks one list of clause compliances corresponding to the clause compliance calculated on the underlying train,valid test split
    """

    def __init__(self,
                 run: int,
                 model,
                 args):
        self.run = run
        self.train_losses = []
        self.train_accuracies = []
        self.valid_losses = []
        self.valid_accuracies = []
        self.test_accuracy = -1
        self.valid_accuracy = -1
        self.epoch_times = []
        self.max_valid_accuracy = -1
        self.max_train_accuracy = -1
        self.avg_epoch_time = -1
        self.eval_step = args.eval_steps
        self.es_enabled = args.es_enabled
        self.es_min_delta = args.es_min_delta
        self.es_patience = args.es_patience
        self.eval_steps = args.eval_steps

        # log knowledge file
        if hasattr(model, 'knowledge'):
            with open(model.knowledge, 'r') as kb_file:
                kb = kb_file.readlines()
            self.knowledge_base = kb
        else:
            self.knowledge_base = None

    def update_per_epoch(self, t_accuracy, v_accuracy, t_loss, v_loss, epoch_time, epoch, model):
        """ collects stats per epoch and updates them """
        self.train_accuracies += [t_accuracy]
        self.valid_accuracies += [v_accuracy]
        self.train_losses += [t_loss]
        self.valid_losses += [v_loss]
        self.epoch_times += [epoch_time]

        if epoch % self.eval_step == 0:
            print(f'Run: {self.run}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {t_loss:.4f}, '
                  f'Time per Train Step: {epoch_time:.6f} '
                  f'Train: {100 * t_accuracy:.2f}%, '
                  f'Valid: {100 * v_accuracy:.2f}% ')

    def update_per_run(self, test_accuracy, valid_accuracy, model):
        """ updates stats per run """
        self.test_accuracy = test_accuracy
        self.valid_accuracy = valid_accuracy
        self.max_valid_accuracy = max(self.valid_accuracies, default=0)
        self.max_train_accuracy = max(self.train_accuracies, default=0)
        self.avg_epoch_time = np.mean(self.epoch_times)

    def to_dict(self):
        return {
            "run": self.run,
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "valid_losses": self.valid_losses,
            "valid_accuracies": self.valid_accuracies,
            "test_accuracy": self.test_accuracy,
            "valid_accuracy": self.valid_accuracy,
            "epoch_times": self.epoch_times,
            "max_valid_accuracy": self.max_valid_accuracy,
            "max_train_accuracy": self.max_train_accuracy,
            "avg_epoch_time": self.avg_epoch_time.data,
            "knowledge_base": self.knowledge_base
        }

    def callback_early_stopping(self, epoch):
        """
        Takes as argument the list with all the validation accuracies.
        If patience=k, checks if the mean of the last k accuracies is higher than the mean of the
        previous k accuracies (i.e. we check that we are not overfitting). If not, stops learning.
        @param valid_accuracies - list(float) , validation accuracy per epoch
        @param epoch: current epoch
        @param args: argument file [Namespace]
        @return bool - if training stops or not
        """
        if not self.es_enabled:
            return False
        else:
            step = len(self.valid_accuracies)
            patience = self.es_patience // self.eval_steps
            # no early stopping for 2 * patience epochs
            if epoch < 2 * self.es_patience:
                return False

            # Mean loss for last patience epochs and second-last patience epochs
            mean_previous = np.mean(self.valid_accuracies[step - 2 * patience:step - patience])
            mean_recent = np.mean(self.valid_accuracies[step - patience:step])
            delta = mean_recent - mean_previous
            if delta <= self.es_min_delta:
                print("*CB_ES* Validation Accuracy didn't increase in the last %d epochs" % patience)
                print("*CB_ES* delta:", delta)
                print(f"callback_early_stopping signal received at epoch {epoch}")
                print("Terminating training")
                return True
            else:
                return False

    def __str__(self):
        return (f"Results of run {self.run}:\n"
                f"Maximum accuracy on train: {self.max_train_accuracy}\n"
                f"Maximum accuracy on valid: {self.max_valid_accuracy}\n"
                f"Accuracy on test: {self.test_accuracy}\n"
                f"Avg epoch time: {self.avg_epoch_time}\n")


class ExperimentLogger(object):
    """
    Experiment statistics per one entry in conf file (at least one or more runs)
    Stores a list of RunStats objects for each run
    """

    def __init__(self, args):
        self.run_stats = []
        self.valid_accuracies = []
        self.test_accuracies = []
        self.train_accuracies = []
        self.avg_epoch_time = -1
        self.avg_train_accuracy = -1
        self.avg_test_accuracy = -1
        self.avg_valid_accuracy = -1
        self.sd_test_accuracy = -1
        self.sd_train_accuracy = -1
        self.sd_valid_accuracy = -1
        self.model = args.model
        self.timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
        self.args = args

    def add_run(self, run: RunLogger):
        self.run_stats.append(run)

    def end_experiment(self):
        """computes experiment statistics over all runs in one conf file """
        self.test_accuracies = [rs.test_accuracy for rs in self.run_stats]
        self.valid_accuracies = [rs.valid_accuracy for rs in self.run_stats]
        self.train_accuracies = [rs.train_accuracies[-1] for rs in self.run_stats]
        self.avg_train_accuracy = np.mean(self.train_accuracies)
        self.avg_test_accuracy = np.mean(self.test_accuracies)
        self.avg_valid_accuracy = np.mean(self.valid_accuracies)
        self.avg_epoch_time = np.mean([rs.avg_epoch_time for rs in self.run_stats])
        if len(self.train_accuracies) > 1:
            self.sd_train_accuracy = np.std(self.train_accuracies)
        if len(self.valid_accuracies) > 1:
            self.sd_valid_accuracy = np.std(self.valid_accuracies)
        if len(self.test_accuracies) > 1:
            self.sd_test_accuracy = np.std(self.test_accuracies)

    def __str__(self):
        runs = len(self.run_stats)
        return (f"Average accuracy over {runs} iterations on valid :{self.avg_valid_accuracy}\n"
                f"Average test accuracy over {runs} iterations :{self.avg_test_accuracy}\n"
                f"Average epoch time: {self.avg_epoch_time}")

