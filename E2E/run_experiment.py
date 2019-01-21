''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

from __future__ import print_function

import argparse
import logging

import torch, os
import torch.utils.data as td

import data_handler
import experiment as ex
import model
import plotter as plt
import trainer

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

logger = logging.getLogger('iCARL')

parser = argparse.ArgumentParser(description='iCarl2.0')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 2.0). Note that lr is decayed by args.gamma parameter args.schedule ')
# parser.add_argument('--schedule', type=int, nargs='+', default=[40, 50, 60],
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20, 30, 40, 50, 60],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1, 0.1, 100, 0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--random-init', action='store_true', default=False,
                    help='Initialize model for next increment using previous weights if false and random weights otherwise')
parser.add_argument('--no-distill', action='store_true', default=False,
                    help='disable distillation loss and only uses the cross entropy loss. See "Distilling Knowledge in Neural Networks" by Hinton et.al for details')
parser.add_argument('--no-random', action='store_true', default=False,
                    help='Disable random shuffling of classes')
parser.add_argument('--no-herding', action='store_true', default=False,
                    help='Disable herding algorithm and do random instance selection instead')
parser.add_argument('--seeds', type=int, nargs='+', default=[1995],
                    help='Seeds values to be used; seed introduces randomness by changing order of classes')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type', default="resnet32",
                    help='model type to be used. Example : resnet32, resnet20, test')
parser.add_argument('--name', default="noname",
                    help='Name of the experiment')
parser.add_argument('--outputDir', default="./",
                    help='Directory to store the results; a new folder "DDMMYYYY" will be created '
                         'in the specified directory to save the results.')
parser.add_argument('--upsampling', action='store_true', default=False,
                    help='Do not do upsampling.')
parser.add_argument('--no-bl', action='store_true', default=False,
                    help='balanced data.')
parser.add_argument('--unstructured-size', type=int, default=0,
                    help='Leftover parameter of an unreported experiment; leave it at 0')
parser.add_argument('--finetune-step', type=int, default=40,
                    help='finetune-step')
parser.add_argument('--alphas', type=float, nargs='+', default=[1.0],
                    help='Weight given to new classes vs old classes in the loss; high value of alpha will increase perfomance on new classes at the expense of older classes. Dynamic threshold moving makes the system more robust to changes in this parameter')
parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')
parser.add_argument('--step-size', type=int, default=10, help='How many classes to add in each increment')
parser.add_argument('--T', type=float, default=2, help='Tempreture used for softening the targets')
parser.add_argument('--memory-budgets', type=int, nargs='+', default=[2000],
                    help='How many images can we store at max. 0 will result in fine-tuning')
parser.add_argument('--epochs-class', type=int, default=70, help='Number of epochs for each increment')
parser.add_argument('--dataset', default="CIFAR100", help='Dataset to be used; example CIFAR, MNIST')
parser.add_argument('--lwf', action='store_true', default=False,
                    help='Use learning without forgetting. Ignores memory-budget '
                         '("Learning with Forgetting," Zhizhong Li, Derek Hoiem)')
parser.add_argument('--no-nl', action='store_true', default=False,
                    help='No Normal Loss. Only uses the distillation loss to train the new model on old classes (Normal loss is used for new classes however')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

dataset = data_handler.DatasetFactory.get_dataset(args.dataset)

# Checks to make sure parameters are sane
if args.step_size < 2:
    print("Step size of 1 will result in no learning;")
    assert False

# Run an experiment corresponding to every seed value
for seed in args.seeds:
    # Run an experiment corresponding to every alpha value
    for at in args.alphas:
        args.alpha = at
        # Run an experiment corresponding to every memory budget
        for m in args.memory_budgets:
            args.memory_budget = m
            # In LwF, memory_budget is 0 (See the paper "Learning without Forgetting" for details).
            if args.lwf:
                args.memory_budget = 0

            # Fix the seed.
            args.seed = seed
            torch.manual_seed(seed)
            if args.cuda:
                torch.cuda.manual_seed(seed)

            # Loader used for training data
            train_dataset_loader = data_handler.IncrementalLoader(dataset.train_data.train_data,
                                                                  dataset.train_data.train_labels,
                                                                  dataset.labels_per_class_train,
                                                                  dataset.classes, [],
                                                                  transform=dataset.train_transform,
                                                                  cuda=args.cuda, oversampling=not args.upsampling,
                                                                  )
            # Special loader use to compute ideal NMC; i.e, NMC that using all the data points to compute the mean embedding
            train_dataset_loader_nmc = data_handler.IncrementalLoader(dataset.train_data.train_data,
                                                                      dataset.train_data.train_labels,
                                                                      dataset.labels_per_class_train,
                                                                      dataset.classes, [],
                                                                      transform=dataset.train_transform,
                                                                      cuda=args.cuda, oversampling=not args.upsampling,
                                                                      )
            # Loader for test data.
            test_dataset_loader = data_handler.IncrementalLoader(dataset.test_data.test_data,
                                                                 dataset.test_data.test_labels,
                                                                 dataset.labels_per_class_test, dataset.classes,
                                                                 [], transform=dataset.test_transform, cuda=args.cuda,
                                                                 )

            kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

            # Iterator to iterate over training data.
            train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
                                                         batch_size=args.batch_size, shuffle=True, **kwargs)
            # Iterator to iterate over all training data (Equivalent to memory-budget = infitie
            train_iterator_nmc = torch.utils.data.DataLoader(train_dataset_loader_nmc,
                                                             batch_size=args.batch_size, shuffle=True, **kwargs)
            # Iterator to iterate over test data
            test_iterator = torch.utils.data.DataLoader(
                test_dataset_loader,
                batch_size=args.batch_size, shuffle=True, **kwargs)

            # Get the required model
            myModel = model.ModelFactory.get_model(args.model_type, args.dataset)
            if args.cuda:
                myModel.cuda()

            # Define an experiment.
            my_experiment = ex.experiment(args.name, args)

            # Adding support for logging. A .log is generated with all the logs. Logs are also stored in a temp file one directory
            # before the code repository
            logger = logging.getLogger('iCARL')
            logger.setLevel(logging.DEBUG)

            fh = logging.FileHandler(my_experiment.path + ".log")
            fh.setLevel(logging.DEBUG)

            fh2 = logging.FileHandler("../temp.log")
            fh2.setLevel(logging.DEBUG)

            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            fh2.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(fh2)
            logger.addHandler(ch)

            # Define the optimizer used in the experiment
            optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                                        weight_decay=args.decay, nesterov=True)

            # Trainer object used for training
            my_trainer = trainer.Trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer,
                                         train_iterator_nmc)

            # Parameters for storing the results
            x = []
            y = []
            y1 = []
            train_y = []
            higher_y = []
            y_scaled = []
            y_grad_scaled = []
            nmc_ideal_cum = []

            # Initilize the evaluators used to measure the performance of the system.
            nmc = trainer.EvaluatorFactory.get_evaluator("nmc", args.cuda)
            nmc_ideal = trainer.EvaluatorFactory.get_evaluator("nmc", args.cuda)
            t_classifier = trainer.EvaluatorFactory.get_evaluator("trainedClassifier", args.cuda, args.T)

            # Loop that incrementally adds more and more classes
            for class_group in range(0, dataset.classes, args.step_size):
                print("SEED:", seed, "MEMORY_BUDGET:", m, "CLASS_GROUP:", class_group)
                # Add new classes to the train, train_nmc, and test iterator
                my_trainer.increment_classes(class_group)
                my_trainer.update_frozen_model(class_group)
                # if class_group:
                #     my_trainer.undate_next_fc(my_trainer.model, class_group//args.step_size)
                epoch = 0
                flag = False


                # Running epochs_class epochs
                # preacc = 0
                # accmax = 0

                for epoch in range(0, args.epochs_class):
                    if class_group and not args.no_bl and epoch >= args.finetune_step and not flag:
                    # if class_group and not args.no_bl:
                        my_trainer.undate_dataloader(my_trainer.train_data_iterator,
                                                     train_dataset_loader,
                                                     args.batch_size, class_group, kwargs
                                                     )
                        my_trainer.tmp_model()
                        flag = True
                    if class_group<10:
                        break
                    my_trainer.update_lr(epoch)
                    my_trainer.train(epoch, class_group)
                    # print(my_trainer.threshold)
                    if epoch % args.log_interval == (args.log_interval - 1):
                        logger.debug("*********CURRENT EPOCH********** : %d", epoch)
                        logger.debug("Train Classifier: %0.2f",
                                     t_classifier.evaluate(class_group, my_trainer.model, train_iterator))
                        logger.debug("Test Classifier: %0.2f",
                                     t_classifier.evaluate(class_group, my_trainer.model, test_iterator))
                        logger.debug("Test Classifier Scaled: %0.2f",
                                     t_classifier.evaluate(class_group, my_trainer.model, test_iterator,
                                                           my_trainer.dynamic_threshold, False,
                                                           my_trainer.older_classes, args.step_size))
                        logger.info("Test Classifier Grad Scaled: %0.2f",
                                    t_classifier.evaluate(class_group, my_trainer.model, test_iterator,
                                                          my_trainer.gradient_threshold_unreported_experiment, False,
                                                          my_trainer.older_classes, args.step_size))

                        # acc_max = max(acc_max, acc_s)
                        # if acc_max > preacc:
                        #     preacc = acc_max
                        #     torch.save(my_trainer.model.state_dict(), 'save_model/cifar100/res32_{}.pkl'.format(class_group))

                # if class_group == 0:
                #     torch.save(my_trainer.model.state_dict(), 'save/task1_10cls_random.pkl')
                # Evaluate the learned classifier
                img = None

                logger.info("Test Classifier Final: %0.2f",
                            t_classifier.evaluate(class_group, my_trainer.model, test_iterator))
                logger.info("Test Classifier Final Scaled: %0.2f",
                            t_classifier.evaluate(class_group, my_trainer.model, test_iterator,
                                                  my_trainer.dynamic_threshold, False,
                                                  my_trainer.older_classes, args.step_size))
                logger.info("Test Classifier Final Grad Scaled: %0.2f",
                            t_classifier.evaluate(class_group, my_trainer.model, test_iterator,
                                                  my_trainer.gradient_threshold_unreported_experiment, False,
                                                  my_trainer.older_classes, args.step_size))

                higher_y.append(
                    t_classifier.evaluate(class_group, my_trainer.model, test_iterator, higher=True))

                y_grad_scaled.append(
                    t_classifier.evaluate(class_group, my_trainer.model, test_iterator,
                                          my_trainer.gradient_threshold_unreported_experiment, False,
                                          my_trainer.older_classes, args.step_size))
                y_scaled.append(t_classifier.evaluate(class_group, my_trainer.model, test_iterator,
                                                      my_trainer.dynamic_threshold, False,
                                                      my_trainer.older_classes, args.step_size))
                y1.append(t_classifier.evaluate(class_group, my_trainer.model, test_iterator))

                # Update means using the train iterator; this is iCaRL case
                nmc.update_means(my_trainer.model, train_iterator, class_group // args.step_size, dataset.classes)
                # Update mean using all the data. This is equivalent to memory_budget = infinity
                nmc_ideal.update_means(my_trainer.model, train_iterator_nmc, class_group // args.step_size,
                                       dataset.classes)
                # Compute the the nmc based classification results
                tempTrain = t_classifier.evaluate(class_group, my_trainer.model, train_iterator)
                train_y.append(tempTrain)

                testY1 = nmc.evaluate(class_group, my_trainer.model, test_iterator, step_size=args.step_size,
                                      kMean=True, )
                testY = nmc.evaluate(class_group, my_trainer.model, test_iterator)
                testY_ideal = nmc_ideal.evaluate(class_group, my_trainer.model, test_iterator)
                y.append(testY)
                nmc_ideal_cum.append(testY_ideal)

                # Compute confusion matrices of all three cases (Learned classifier, iCaRL, and ideal NMC)
                # tcMatrix = t_classifier.get_confusion_matrix(class_group,my_trainer.model, test_iterator, dataset.classes)
                # tcMatrix_scaled = t_classifier.get_confusion_matrix(class_group,my_trainer.model, test_iterator, dataset.classes,
                #                                                     my_trainer.dynamic_threshold, my_trainer.older_classes,
                #                                                     args.step_size)
                # tcMatrix_grad_scaled = t_classifier.get_confusion_matrix(class_group,my_trainer.model, test_iterator,
                #                                                          dataset.classes,
                #                                                          my_trainer.gradient_threshold_unreported_experiment,
                #                                                          my_trainer.older_classes,
                #                                                          args.step_size)
                # nmcMatrix = nmc.get_confusion_matrix(class_group, my_trainer.model, test_iterator, dataset.classes)
                # nmcMatrixIdeal = nmc_ideal.get_confusion_matrix(class_group,my_trainer.model, test_iterator, dataset.classes)
                # tcMatrix_scaled_binning = t_classifier.get_confusion_matrix(class_group,my_trainer.model, test_iterator,
                #                                                             dataset.classes,
                #                                                             my_trainer.dynamic_threshold,
                #                                                             my_trainer.older_classes,
                #                                                             args.step_size, True)

                my_trainer.setup_training()

                # Store the resutls in the my_experiment object; this object should contain all the information required to reproduce the results.
                x.append(class_group + args.step_size)

                my_experiment.results["NMC"] = [x, [float(p) for p in y]]
                my_experiment.results["Trained Classifier"] = [x, [float(p) for p in y1]]
                my_experiment.results["Trained Classifier Scaled"] = [x, [float(p) for p in y_scaled]]
                my_experiment.results["Trained Classifier Grad Scaled"] = [x, [float(p) for p in y_grad_scaled]]
                my_experiment.results["Train Error Classifier"] = [x, [float(p) for p in train_y]]
                my_experiment.results["Ideal NMC"] = [x, [float(p) for p in nmc_ideal_cum]]
                my_experiment.store_json()

                # Finally, plotting the results;
                # my_plotter = plt.Plotter()
                #
                # # Plotting the confusion matrices
                # my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                #                       my_experiment.path + "tcMatrix", tcMatrix)
                # my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                #                       my_experiment.path + "tcMatrix_scaled", tcMatrix_scaled)
                # my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                #                       my_experiment.path + "tcMatrix_scaled_binning", tcMatrix_scaled_binning)
                # my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                #                       my_experiment.path + "nmcMatrix",
                #                       nmcMatrix)
                # my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                #                       my_experiment.path + "nmcMatrixIdeal",
                #                       nmcMatrixIdeal)
                #
                # # Plotting the line diagrams of all the possible cases
                # my_plotter.plot(x, y, title=args.name, legend="NMC")
                # my_plotter.plot(x, higher_y, title=args.name, legend="Higher Model")
                # my_plotter.plot(x, y_scaled, title=args.name, legend="Trained Classifier Scaled")
                # my_plotter.plot(x, y_grad_scaled, title=args.name, legend="Trained Classifier Grad Scaled")
                # my_plotter.plot(x, nmc_ideal_cum, title=args.name, legend="Ideal NMC")
                # my_plotter.plot(x, y1, title=args.name, legend="Trained Classifier")
                # my_plotter.plot(x, train_y, title=args.name, legend="Trained Classifier Train Set")
                #
                # # Saving the line plot
                # my_plotter.save_fig(my_experiment.path, dataset.classes + 1)

