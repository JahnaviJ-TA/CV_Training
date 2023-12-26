# import the necessary packages
from pyimagesearch.utils_map import run_inference
from pyimagesearch.utils_map import load_yolo_cls_idx
from pyimagesearch import config
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from darknet import darknet

def compute_map():
	# use the COCO class to load and read the ground-truth annotations
	cocoAnnotation = COCO(annotation_file=config.COCO_GT_ANNOTATION)
	
	# call the darknet.load_network method, which loads the YOLOv4
	# network based on the configuration, weights, and data file
	(network, classNames, _) = darknet.load_network(
			config.YOLO_CONFIG,
			config.COCO_DATA,
			config.YOLO_WEIGHTS,
		)
	label2Idx = load_yolo_cls_idx(config.LABEL2IDX)
	yoloCls90 = load_yolo_cls_idx(config.YOLO_90CLASS_MAP)
	
  # call the run_inference function to generate prediction JSON file
	run_inference(config.IMAGES_PATH, network, classNames, label2Idx,
		yoloCls90, config.CONF_THRESHOLD, config.COCO_VAL_PRED)	
  
  # load detection JSON file from the disk
	cocovalPrediction = cocoAnnotation.loadRes(config.COCO_VAL_PRED)
	# initialize the COCOeval object by passing the coco object with
	# ground truth annotations, coco object with detection results
	cocoEval = COCOeval(cocoAnnotation, cocovalPrediction, "bbox")
	
	# run evaluation for each image, accumulates per image results
	# display the summary metrics of the evaluation
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()