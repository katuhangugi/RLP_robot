build-image:
	docker build -t rpl-gpu:test-0.1 .

start-container:
	docker run --ulimit nofile=655350:655350 --privileged --cap-add SYS_ADMIN --device /dev/fuse --ulimit memlock=-1 --cpu-shares=15360 --cpu-quota=9600000 --cpu-period=100000 --memory=500000m -itd --net=host -v $(CURDIR):/workspace/code -v /home/joe/Work/RPL_local/logs:/workspace/code/logs --gpus all --device=/dev/nvidiactl:/dev/nvidiactl --name=rpl-test rpl-gpu:test-0.1 bash
	docker exec -it rpl-test bash

stop-container:
	docker stop rpl-test

rm-container: stop-container
	docker rm rpl-test

clone-code:
	git clone -b master https://ghp.ci/https://github.com/k-r-allen/residual-policy-learning.git
	mkdir residual-policy-learning/src
	git clone -b tf2 https://ghp.ci/https://github.com/openai/baselines.git residual-policy-learning/src/baselines
	rm -rf residual-policy-learning/src/baselines/.git

patch-code:
	cp *.patch residual-policy-learning
	cd residual-policy-learning && echo $(pwd) && git apply 0001-baselines.patch
	cd residual-policy-learning && git apply 0002-rpl.patch
	cd residual-policy-learning && git apply 0003-.patch

#conda env create -f py10.yaml -n py10-2

install-rpl:
	cd residual-policy-learning/rpl_environments && pip install -e .

install-baselines:
	cd residual-policy-learning/src/baselines && pip install -e .

deps: install-rpl install-baselines

test-train:
	cd residual-policy-learning/tensorflow/experiment && python3 train_staged.py --env FetchPickAndPlace-v1 --n_epochs 100 --num_cpu 1 --config_path=configs/pickandplace.json --logdir /workspace/code/logs/seed0/FetchPickAndPlace-v1 --seed 0 --random_eps=0.3 --controller_prop=0.8

test-all:
	ln -sf /usr/bin/python3 /usr/bin/python && \
	cd residual-policy-learning/tensorflow/experiment && \
	chmod +x run_all_experiments.sh && \
	./run_all_experiments.sh


