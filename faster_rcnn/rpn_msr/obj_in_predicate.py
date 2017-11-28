import torch

def obj_in_predicate(object_rois, region_rois):
	num_obj = object_rois.size(0)
	num_rel = region_rois.size(0)
	repeat_region = region_rois.repeat(1, num_obj).view(-1, 4)
	repeat_obj = num_obj.repeat(num_rel, 1)
	index = (repeat_obj[:, 0] >= repeat_region[:, 0]).data & \
			(repeat_obj[:, 1] >= repeat_region[:, 1]).data & \
			(repeat_obj[:, 2] <= repeat_region[:, 2]).data & \
			(repeat_obj[:, 3] <= repeat_region[:, 3]).data
	index = index.nonzero()