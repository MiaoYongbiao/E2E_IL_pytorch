''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

from __future__ import print_function

import copy
import logging

import numpy as np
import torch, random
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler

import model

logger = logging.getLogger('iCARL')


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this.
    '''

    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator=None):
        self.train_data_iterator = trainDataIterator
        self.test_data_iterator = testDataIterator
        self.model = model
        self.args = args
        self.dataset = dataset
        self.train_loader = self.train_data_iterator.dataset
        self.older_classes = []
        self.optimizer = optimizer
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed_tmp = copy.deepcopy(self.model)
        self.active_classes = []
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.models = []
        self.current_lr = args.lr
        self.all_classes = list(range(dataset.classes))
        self.all_classes.sort(reverse=True)
        self.left_over = []
        self.ideal_iterator = ideal_iterator
        self.model_single = copy.deepcopy(self.model)
        self.optimizer_single = None

        logger.warning("Shuffling turned off for debugging")
        # random.seed(args.seed)
        # random.shuffle(self.all_classes)


class Trainer(GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator=None):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator)
        self.dynamic_threshold = np.ones(self.dataset.classes // self.args.step_size, dtype=np.float64)
        # self.dynamic_threshold = np.ones(self.dataset.classes, dtype=np.float64)
        self.gradient_threshold_unreported_experiment = np.ones(self.dataset.classes // self.args.step_size,
                                                                dtype=np.float64)
        # self.gradient_threshold_unreported_experiment = np.ones(self.dataset.classes,
        #                                                         dtype=np.float64)

    def update_lr(self, epoch):
        for temp in range(0, len(self.args.schedule)):
            if self.args.schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    logger.debug("Changing learning rate from %0.4f to %0.4f", self.current_lr,
                                 self.current_lr * self.args.gammas[temp])
                    self.current_lr *= self.args.gammas[temp]

    def increment_classes(self, class_group):
        '''
        Add classes starting from class_group to class_group + step_size
        :param class_group:
        :return: N/A. Only has side-affects
        '''
        for temp in range(class_group, class_group + self.args.step_size):
            pop_val = self.all_classes.pop()
            self.train_data_iterator.dataset.add_class(pop_val)
            self.ideal_iterator.dataset.add_class(pop_val)
            self.test_data_iterator.dataset.add_class(pop_val)
            self.left_over.append(pop_val)

    def limit_class(self, n, k, herding=True):
        if not herding:
            self.train_loader.limit_class(n, k)
        else:
            self.train_loader.limit_class_and_sort(n, k, self.model_fixed)
        if n not in self.older_classes:
            self.older_classes.append(n)

    def reset_dynamic_threshold(self):
        '''
        Reset the threshold vector maintaining the scale factor.
        Important to set this to zero before every increment.
        setupTraining() also does this so not necessary to call both.
        :return:
        '''
        threshTemp = self.dynamic_threshold / np.max(self.dynamic_threshold)
        threshTemp = ['{0:.4f}'.format(i) for i in threshTemp]

        threshTemp2 = self.gradient_threshold_unreported_experiment / np.max(
            self.gradient_threshold_unreported_experiment)
        threshTemp2 = ['{0:.4f}'.format(i) for i in threshTemp2]

        logger.debug("Scale Factor" + ",".join(threshTemp))
        logger.debug("Scale GFactor" + ",".join(threshTemp2))

        self.dynamic_threshold = np.ones(self.args.step_size + len(self.dynamic_threshold), dtype=np.float64)
        # self.dynamic_threshold = np.ones(self.dataset.classes, dtype=np.float64)
        self.gradient_threshold_unreported_experiment = np.ones(len(self.dynamic_threshold),
                                                                dtype=np.float64)
        # self.gradient_threshold_unreported_experiment = np.ones(self.dataset.classes,
        #                                                         dtype=np.float64)

    def setup_training(self):
        self.reset_dynamic_threshold()

        for param_group in self.optimizer.param_groups:
            logger.debug("Setting LR to %0.2f", self.args.lr)
            param_group['lr'] = self.args.lr
            self.current_lr = self.args.lr
        for val in self.left_over:
            self.limit_class(val, int(self.args.memory_budget / len(self.left_over)), not self.args.no_herding)

    def update_frozen_model(self, class_group):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.models.append(self.model_fixed)

        if class_group==0:
            self.model.load_state_dict(torch.load('save/task1_10cls_random.pkl'))
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                             weight_decay=self.args.decay, nesterov=True)
            self.model.eval()

        if self.args.random_init:
            logger.warning("Random Initilization of weights at each increment")
            myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)
            if self.args.cuda:
                myModel.cuda()
            self.model = myModel
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                             weight_decay=self.args.decay, nesterov=False)
            self.model.eval()

    def randomly_init_model(self):
        logger.info("Randomly initilizaing model")
        myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)
        if self.args.cuda:
            myModel.cuda()
        self.model = myModel
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                         weight_decay=self.args.decay, nesterov=True)
        self.model.eval()

    def get_model(self):
        myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)
        if self.args.cuda:
            myModel.cuda()
        optimizer = torch.optim.SGD(myModel.parameters(), self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.decay, nesterov=True)
        myModel.eval()

        self.current_lr = self.args.lr

        self.model_single = myModel
        self.optimizer_single = optimizer

    def make_weights_for_balanced_classes(self, dic, class_group):
        nclasses = len(dic)
        count = [0] * nclasses
        for key, (start, end) in dic.items():
            if key < class_group + self.args.step_size:
                count[key] = end - start
        weight_per_class = [0.] * nclasses
        # N = float(sum(count))
        N = (sum(count))
        for i in range(nclasses):
            if count[i] == 0:
                weight_per_class[i] = 0
            else:
                weight_per_class[i] = N / float(count[i])
        weight = [0] * (sum(count[:class_group]) + 5000)
        # for idx, val in enumerate(images):
        weight[:sum(count[:class_group])] = [weight_per_class[0]] * sum(count[:class_group])
        weight[sum(count[:class_group]):] = [weight_per_class[class_group]] * 5000

        return weight

    def undate_next_fc(self, model, ind):
        self.model.eval()
        self.model.fc_lst[ind].load_state_dict(self.models[-1].fc_lst[ind - 1].state_dict())
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr / 10, momentum=self.args.momentum,
                                         weight_decay=self.args.decay, nesterov=True)
        self.model.eval()

    def limit_class_finetune(self, n, k, herding=True):
        if not herding:
            self.train_loader.limit_class(n, k)
        else:
            self.train_loader.limit_class_and_sort(n, k, self.model_fixed)
        # if n not in self.older_classes:
        #     self.older_classes.append(n)

    def undate_dataloader(self, train_data_iterator, train_dataset_loader, batch_size, class_group, kwargs):
        for val in self.left_over:
            self.limit_class_finetune(val, int(self.args.memory_budget / (len(self.left_over)-self.args.step_size)), not self.args.no_herding)
        # weights = self.make_weights_for_balanced_classes(train_data_iterator.dataset.indices, class_group)
        # train_sampler = WeightedRandomSampler(weights,
        #                                       num_samples=2000, replacement=True)
        # train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
        #                                              batch_size=batch_size,
        #                                              sampler=train_sampler,
        #                                              shuffle=False, **kwargs)
        # self.train_data_iterator = train_iterator

    def tmp_model(self):
        self.model_fixed_tmp = copy.deepcopy(self.model)
        self.model_fixed_tmp.eval()
        for param in self.model_fixed_tmp.parameters():
            param.requires_grad = False

    def train(self, epoch, class_group):

        self.model.train()
        logger.info("Epochs %d", epoch)
        incremental_ind = class_group // self.args.step_size
        cls_total = class_group + self.args.step_size
        for data, y, target in tqdm(self.train_data_iterator):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
                y = y.cuda()
            oldClassesIndices = (target * 0).int()
            for elem in range(0, self.args.unstructured_size):
                oldClassesIndices = oldClassesIndices + (target == elem).int()

            old_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices > 0)).long())
            new_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices == 0)).long())

            self.optimizer.zero_grad()

            target_normal_loss = target[new_classes_indices]
            data_normal_loss = data[new_classes_indices]

            target_distillation_loss = y.float()
            data_distillation_loss = data

            y_onehot = torch.FloatTensor(len(target_normal_loss), self.dataset.classes)
            if self.args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target_normal_loss.unsqueeze_(1)
            y_onehot.scatter_(1, target_normal_loss, 1)

            if not class_group:
                output = self.model(Variable(data_normal_loss), class_group // self.args.step_size)
                # loss = F.kl_div(output, Variable(y_onehot[:, :class_group + self.args.step_size]))
                loss = F.nll_loss(output, Variable(target))
                self.dynamic_threshold += np.sum(y_onehot[:, :self.args.step_size].cpu().numpy(), 0)

            myT = self.args.T

            if self.args.no_distill:
                pass

            elif len(self.older_classes) > 0:

                loss_lst = []
                dis_lst = []
                # pred3_lst = torch.zeros(len(target), 100, dtype=torch.float32).cuda()
                pred3_lst = []
                output3_lst = []
                for task_ind in range(incremental_ind):
                    # pred3 = self.models[task_ind+1](Variable(data), task_ind, T=myT, labels=True).data
                    pred3 = self.model_fixed(Variable(data), task_ind, T=myT, labels=True).data
                    # pred3_lst.append(pred3)
                    # pred3_lst[:, task_ind*self.args.step_size:(task_ind+1)*self.args.step_size] += pred3
                    # pred3 = self.model(Variable(data), task_ind, T=myT, labels=True).data
                    output3 = self.model(Variable(data), task_ind, T=myT)
                    # output3 = self.model(Variable(data), task_ind, T=myT, logit=True)
                    # output3_lst.append(output3)
                    # for cls_ind in range(self.args.step_size, class_group + self.args.step_size, self.args.step_size):
                    # y = torch.zeros(len(target), cls_total).cuda()
                    # pred3 = self.models[cls_ind//self.args.step_size](Variable(data), incremental_ind, T=myT, labels=True).data
                    # pred3_g = pred3[:, :cls_total] * y.copy_(
                    #     torch.exp(output3[:, :cls_total]).sum(dim=1).reshape(len(target), 1))

                    loss3 = F.kl_div(output3, Variable(pred3))
                    # loss3 = F.kl_div(torch.log(output3), Variable(pred3))
                    # loss3 = -(output3 * Variable(pred3)).sum()/len(target)
                    # loss3 = F.kl_div(output3[:, cls_ind-self.args.step_size:cls_ind],
                    #                  Variable(pred3[:,  cls_ind-self.args.step_size:cls_ind]))
                    # loss3 = F.kl_div(output3[:, task_ind*self.args.step_size:(task_ind+1)*self.args.step_size],
                    #                  Variable(pred3[:, task_ind*self.args.step_size:(task_ind+1)*self.args.step_size]))
                    # dis_lst.append(loss3.data[0])
                    # loss3.backward(retain_graph=True)
                    # pred3_lst += pred3
                    dis_lst.append(loss3)

                    loss3.backward(retain_graph=True)

                if epoch >= 40:
                    pred3 = self.model_fixed_tmp(Variable(data), incremental_ind, T=myT, labels=True).data
                    output3 = self.model(Variable(data), incremental_ind, T=myT)
                    loss3 = F.kl_div(output3, Variable(pred3))
                    dis_lst.append(loss3)
                    loss3.backward(retain_graph=True)

                # pred3 = torch.sum(torch.stack(pred3_lst), dim=1)
                # for i in range(0, class_group + 1, self.args.step_size):
                #     pred3_lst[:, i:i + self.args.step_size] = pred3_lst[:, i:i + self.args.step_size] / (
                #                 (class_group - i) // self.args.step_size + 1)
                pred3 = self.model_fixed(Variable(data), incremental_ind, T=myT, logit=True)
                # pred3_lst = torch.cat(pred3_lst, dim=1)
                # pred3_lst /= incremental_ind
                # self.dynamic_threshold[:cls_total - self.args.step_size] += \
                #     (np.sum(pred3_lst.data.cpu().numpy(), 0)) * (
                #             myT * myT) * self.args.alpha
                self.dynamic_threshold[:cls_total] += \
                    (np.sum(F.softmax(pred3, dim=1).data.cpu().numpy(), 0)) * (
                            myT * myT) * self.args.alpha


                output_total = torch.zeros(len(target), 100, dtype=torch.float32).cuda()
                # output_total = []
                # for task_ind in range(incremental_ind + 1):
                # if task_ind != class_group // self.args.step_size: continue
                # previous_old = target < task_ind * self.args.step_size
                # old_data_ind_t = target < (task_ind + 1) * self.args.step_size
                # old_data_ind = old_data_ind_t - previous_old
                # if sum(old_data_ind.data) == 0: continue
                output_total = self.model(Variable(data), incremental_ind, logit=True)
                # output_total.append(output)
                # output_total = torch.cat(output_total, dim=1)
                # output = self.model(Variable(data[old_data_ind]), task_ind, per_logit=True)
                # output_total[old_data_ind, task_ind*self.args.step_size:(task_ind+1)*self.args.step_size] += output
                # output_total = F.log_softmax(output_total, dim=1)
                # output = self.model(Variable(data), incremental_ind)
                loss = F.kl_div(F.log_softmax(output_total, dim=1), Variable(y_onehot[:, :cls_total]))
                # loss = F.nll_loss(F.log_softmax(output_total, dim=1), Variable(target))
                loss_lst.append(loss)
                # if task_ind != class_group // self.args.step_size:
                #     loss.backward(retain_graph=True)
                # else:
                #     loss.backward()
                self.dynamic_threshold += np.sum(y_onehot[:,:cls_total].cpu().numpy(), 0)

                # Scale gradient by a factor of square of T. See Distilling Knowledge in Neural Networks by Hinton et.al. for details.
                # if class_group:
                #     loss3 = torch.sum(torch.stack(dis_lst))/incremental_ind
                #     loss3.backward(retain_graph=True)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad * (myT * myT) * self.args.alpha
                # for param in self.model.named_parameters():
                #     if "fc" in param[0] and 'weight' in param[0] and int(
                #             param[0][param[0].find(".") - 1]) > class_group // self.args.step_size - 1:
                #         break
                #     elif param[1].grad is not None:
                #         param[1].grad = param[1].grad * (myT * myT) * self.args.alpha

            # sigma = 0.3 / (1 + epoch) ** 0.55
            # for param in self.model.named_parameters():
            #     if 'weight' in param[0]:
            #         param[1] += sigma*Variable(torch.randn(param[1].shape).cuda())

            if len(self.older_classes) == 0 or not self.args.no_nl:
                # if not class_group or not self.args.no_nl:
                # if class_group:
                #     loss3 = torch.sum(torch.stack(dis_lst))/incremental_ind
                #     loss3.backward(retain_graph=True)
                loss = torch.sum(torch.stack(loss_lst))
                loss.backward()

            # sigma = 0.3 / ((1 + epoch) ** 0.55)
            # for param in self.model.named_parameters():
            #     if 'weight' in param[0] and 'conv' in param[0] or 'fc' in param[0] and param[1].grad is not None:
            #         # param[1].grad += sigma*Variable(torch.randn(param[1].grad.shape).cuda())
            #         param[1].grad += 1e-2 * torch.normal(mean=torch.zeros(param[1].shape),
            #                                   std=(torch.ones(param[1].shape) * 0.3 / (1 + epoch) ** 0.55) ** 0.5).cuda()

            for param in self.model.named_parameters():
                # if "fc{}.weight".format(class_group // self.args.step_size) in param[0]:
                if "fc" in param[0] and 'weight' in param[0]:
                    ind = int(param[0][param[0].find(".") - 1])
                    if ind <= class_group // self.args.step_size:
                        self.gradient_threshold_unreported_experiment[ind*self.args.step_size:(ind+1)*self.args.step_size] += np.sum(
                            np.abs(param[1].grad.data.cpu().numpy()), 1)
            self.gradient_threshold_unreported_experiment *= 0.99
            # for param in self.model.named_parameters():
            #     if "fc0.weight" in param[0]:
            #         self.gradient_threshold_unreported_experiment *= 0.99
            #         self.gradient_threshold_unreported_experiment += np.sum(np.abs(param[1].grad.data.cpu().numpy()), 1)

            self.optimizer.step()

        if len(self.older_classes) == 0:
            print('loss is :{:.6f}\t'.format(loss.data))
            # print('loss is :{:.6f}\t mmd is :{:.6f}\t'.format(loss.data, loss2.data))
            # print('loss is :{:.6f}\t distill_3 is : {:.6f}'.format(loss.data, loss3.data))
            # print('loss is :{}\t distill_total is :{}\t distill_per is : {}'.format(loss.data, loss2.data, loss3.data))
        else:
            # print('loss is :{:.6f}\t'.format(loss.data))
            print('loss is :{:.6f}\t loss3 is :{:.6f}\t'.format(loss.data[0], loss3.data))
            # print('loss is :{:.6f}\tdistill_cur is :{:.6f}\tdistill_old is : {:.6f}'.format(loss.data, loss2.data, loss3.data))

        if self.args.no_nl:
            self.dynamic_threshold[len(self.older_classes):len(self.dynamic_threshold)] = np.max(self.dynamic_threshold)
            self.gradient_threshold_unreported_experiment[
            len(self.older_classes):len(self.gradient_threshold_unreported_experiment)] = np.max(
                self.gradient_threshold_unreported_experiment)
        else:
            self.dynamic_threshold[0:self.args.unstructured_size] = np.max(self.dynamic_threshold)
            self.gradient_threshold_unreported_experiment[0:self.args.unstructured_size] = np.max(
                self.gradient_threshold_unreported_experiment)

            self.dynamic_threshold[self.args.unstructured_size + len(
                self.older_classes) + self.args.step_size: len(self.dynamic_threshold)] = np.max(
                self.dynamic_threshold)
            self.gradient_threshold_unreported_experiment[self.args.unstructured_size + len(
                self.older_classes) + self.args.step_size: len(self.gradient_threshold_unreported_experiment)] = np.max(
                self.gradient_threshold_unreported_experiment)

    def add_model(self):
        model = copy.deepcopy(self.model_single)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        logger.debug("Total Models %d", len(self.models))
