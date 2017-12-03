import os
import torch
import numpy as np
import time
from faster_rcnn import network
from faster_rcnn.RPN import RPN  # Hierarchical_Descriptive_Model
from faster_rcnn.utils.timer import Timer
from faster_rcnn.utils.HDN_utils import check_recall

from faster_rcnn.datasets.visual_genome_loader import visual_genome
from faster_rcnn.fast_rcnn.config import cfg
import argparse

import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser('Options for training RPN in pytorch')

## training settings
parser.add_argument('--lr', type=float, default=0.01, help='To disable the Lanuage Model ')
parser.add_argument('--max_epoch', type=int, default=5, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--log_interval', type=int, default=500, help='Interval for Logging')
parser.add_argument('--disable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')
parser.add_argument('--step_size', type=int, default=2, help='step to decay the learning rate')

## Environment Settings
parser.add_argument('--pretrained_model', type=str, default='model/pretrained_models/VGG_imagenet.npy',
                    help='Path for the to-evaluate model')
parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output/RPN', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='RPN_relationship', help='model name for snapshot')
parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
args = parser.parse_args()


def main():
	global args
	print "Loading training set and testing set..."
	# train_set = visual_genome(args.dataset_option, 'train')
	test_set = visual_genome('small', 'test')
	print "Done."

	# train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
	net = RPN(args.use_normal_anchors)
	network.load_net('./output/RPN/RPN_region_best.h5', net)
	# network.set_trainable(net.features, requires_grad=False)
	net.cuda()

	best_recall = np.array([0.0, 0.0])


	# Testing
	recall = test(test_loader, net)

	print('Recall: '
	      'object: {recall[0]: .3f}%% (Best: {best_recall[0]: .3f}%%)'
	      'relationship: {recall[1]: .3f}%% (Best: {best_recall[1]: .3f}%%)'.format(
		recall=recall*100, best_recall=best_recall*100))


def test(test_loader, target_net):
	box_num = np.array([0, 0])
	correct_cnt, total_cnt = np.array([0, 0]), np.array([0, 0])
	print '========== Testing ======='
	target_net.eval()

	batch_time = network.AverageMeter()
	end = time.time()
	for i, (im_data, im_info, gt_objects, gt_relationships, gt_boxes_relationship) in enumerate(test_loader):
		correct_cnt_t, total_cnt_t = np.array([0, 0]), np.array([0, 0])
		# Forward pass
		object_rois, relationship_rois = target_net(im_data, im_info.numpy(), gt_objects.numpy(),
		                                            gt_boxes_relationship.numpy())[1:]
		box_num[0] += object_rois.size(0)
		box_num[1] += relationship_rois.size(0)
		correct_cnt_t[0], total_cnt_t[0] = check_recall(object_rois, gt_objects[0].numpy(), 1000, thres=0.5)
		correct_cnt_t[1], total_cnt_t[1] = check_recall(relationship_rois, gt_boxes_relationship[0].numpy(), 1000, thres=0.6)
		correct_cnt += correct_cnt_t
		total_cnt += total_cnt_t
		batch_time.update(time.time()-end)
		end = time.time()
		if (i+1)%100 == 0 and i > 0:
			print('[{0}/{10}]  Time: {1:2.3f}s/img).'
			      '\t[object] Avg: {2:2.2f} Boxes/im, Top-1000 recall: {3:2.3f} ({4:d}/{5:d})'
			      '\t[relationship] Avg: {6:2.2f} Boxes/im, Top-1000 recall: {7:2.3f} ({8:d}/{9:d})'.format(
				i+1, batch_time.avg,
				box_num[0]/float(i+1), correct_cnt[0]/float(total_cnt[0])*100, correct_cnt[0], total_cnt[0],
				box_num[1]/float(i+1), correct_cnt[1]/float(total_cnt[1])*100, correct_cnt[1], total_cnt[1],
				len(test_loader)))

	recall = correct_cnt/total_cnt.astype(np.float)
	print '====== Done Testing ===='
	return recall


if __name__ == '__main__':
	main()
