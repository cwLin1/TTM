# TTM

Data Preprocessing (Video to audio)

	python utils/data_preprocess.py
	

Data Preprocessing (Video to feature)
	
	cd video_features
	
	# install environment
	conda env create -f conda_env_torch_zoo.yml

	# load the environment
	conda activate torch_zoo

	python main.py feature_type=i3d device="cuda:0" stream=["rgb"]

Download Feature
-I3D: https://drive.google.com/file/d/1hbdqAQhNR4I7vuxrVDG385LjNPA4hkbZ/view?usp=share_link 
-MFCC: https://drive.google.com/file/d/1EzXAI36-1XvS2WhVxzoGo_AREWXxBTxQ/view?usp=share_link 

Dataset 

	python utils/datasets.py

Data Format 

	TTM / 
		dlcv-final-problem1-talking-to-me/student_data/student_data/
			audios/
			i3d/
			MFCC/
			train/
			test/
			videos/
		utils/ 
			dataset.py 
			data_preprocess.py 
		models/ 
			model.py 
		train.py 
		eval.py 
