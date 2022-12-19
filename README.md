# TTM

Data Preprocessing (Video to audio)

	python utils/data_preprocess.py
	

Data Preprocessing (Video to feature)
	
	cd video_features
	python main.py feature_type=i3d device="cuda:0" stream=["rgb"]

Dataset 

	python utils/datasets.py

Data Format 

	TTM / 
		dlcv-final-problem1-talking-to-me/… 
		utils/ 
			dataset.py 
			data_preprocess.py 
		models/ 
			model.py 
		train.py 
		eval.py 
