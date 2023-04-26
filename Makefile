onnx-export:
	python export.py --weights yolo.pt --grid --end2end --simplify \
	--topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640

tensorflow-export:
	./.venv/bin/onnx-tf convert -i yolo.onnx -o ./tensorflow-model/

# requires tensorflow-export to have been run first
tflite-export:
	python export_tflite.py ./tensorflow-model -o ./yolo.tflite --verify_tflite ./test.jpg