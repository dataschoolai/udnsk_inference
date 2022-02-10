input_dir=$workspace/'data/104'
output_dir=$workspace/'output/104'
model_dir=$workspace/'yolo_model/Exposure_100/weights/best.pt'
python detect.py --input_dir $input_dir --output_dir $output_dir --model_dir $model_dir